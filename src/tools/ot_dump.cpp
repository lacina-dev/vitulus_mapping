// ot_dump — dump ALL leaf centres of an octomap .ot to a flat binary file.
//
// Mapping v3 / classifier v2 (2026-07-19). band_projector needs octomap FREE
// leaves (the actually-raytraced free space) to build an honest FREE mask
// (band.project_band D1). octomap_server only publishes OCCUPIED centres on the
// wire, so we read the saved garden.ot off disk with this helper instead.
//
//   usage: ot_dump in.ot out.bin [no_expand]
//
// Output: little-endian float32 records [x, y, z, size, occ] per leaf, where
// occ = 1.0 for occupied, 0.0 for free. By default the tree is expand()-ed so
// pruned free space is emitted at full leaf resolution (pass a 3rd arg "0" or
// "no_expand" to skip). Progress/summary is printed to stderr; stdout is empty.
//
// Derived from the forensic harness dump_all.cpp (reused per the classifier v2
// design). Built by CMake (add_executable) into
// devel/lib/vitulus_mapping/ot_dump, callable via rosrun vitulus_mapping ot_dump
// or by absolute devel path.
#include <octomap/octomap.h>
#include <octomap/OcTree.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>

using namespace octomap;

int main(int argc, char** argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: ot_dump in.ot out.bin [no_expand]\n");
    return 1;
  }
  bool expand = true;
  if (argc > 3) {
    const char* a = argv[3];
    if (strcmp(a, "0") == 0 || strcmp(a, "no_expand") == 0 ||
        strcmp(a, "false") == 0)
      expand = false;
  }
  AbstractOcTree* a = AbstractOcTree::read(argv[1]);
  if (!a) { fprintf(stderr, "ot_dump: read failed: %s\n", argv[1]); return 2; }
  OcTree* t = dynamic_cast<OcTree*>(a);
  if (!t) { fprintf(stderr, "ot_dump: not an OcTree: %s\n", argv[1]); return 3; }
  if (expand) t->expand();  // full-res free space
  FILE* out = fopen(argv[2], "wb");
  if (!out) { fprintf(stderr, "ot_dump: cannot open %s\n", argv[2]); return 4; }
  long occ = 0, freen = 0;
  double res = t->getResolution();
  for (OcTree::leaf_iterator it = t->begin_leafs(), end = t->end_leafs();
       it != end; ++it) {
    float o = t->isNodeOccupied(*it) ? 1.f : 0.f;
    if (o > 0.5f) occ++; else freen++;
    float rec[5] = { (float)it.getX(), (float)it.getY(), (float)it.getZ(),
                     (float)it.getSize(), o };
    fwrite(rec, sizeof(float), 5, out);
  }
  fclose(out);
  delete t;
  fprintf(stderr, "ot_dump: occ=%ld free=%ld res=%.3f -> %s\n",
          occ, freen, res, argv[2]);
  return 0;
}
