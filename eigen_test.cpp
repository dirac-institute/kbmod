#include<iostream>
#include <cmath>
#include<array>
#include<Eigen/Core>
#include <chrono>



struct RealCoord;


struct Index {
  int x;
  int y;
  Index(int i, int j)
    : x(i), y(j) {}
  Index(double xd, double yd)
    : x(floor(xd)), y(floor(yd)) {}

  Index(float xf, double yf)
    : x(floor(xd)), y(floor(yd)) {}

  //Index(struct RealCoord *p)
  //  : x(floor(p->x)), y(floor(p->y)) {}
  friend std::ostream& operator<<(std::ostream& os, const Index& rc);
};

struct RealCoord{
  double x;
  double y;
  RealCoord(double xd, double yd)
    : x(xd), y(yd) {}
  std::array<RealCoord, 4> nearest_neighbors_coords(){
    return {
      RealCoord(floor(x-0.5)+0.5, floor(y+0.5)+0.5),
      RealCoord(floor(x+0.5)+0.5, floor(y+0.5)+0.5),
      RealCoord(floor(x-0.5)+0.5, floor(y-0.5)+0.5),
      RealCoord(floor(x+0.5)+0.5, floor(y-0.5)+0.5)
    };
  };
  std::array<Index, 4>  nearest_neighbors_idxs(){
    return {
      Index(x-0.5, y+0.5),
      Index(x+0.5, y+0.5),
      Index(x-0.5, y-0.5),
      Index(x+0.5, y-0.5)
    };
  };
  friend std::ostream& operator<<(std::ostream& os, const RealCoord& rc);
};


struct PixCoord {
  int x;
  int y;
  PixCoord(double xd, double yd)
    : x(floor(xd)), y(floor(yd)) {}
  PixCoord(RealCoord p)
    : x(floor(p.x)+0.5), y(floor(p.y)+0.5) {}
  friend std::ostream& operator<<(std::ostream& os, const PixCoord& rc);
};


std::ostream& operator<<(std::ostream& os, const Index& rc){
  os << "x: " << rc.x << " y: " << rc.y;
  return os;
}


std::ostream& operator<<(std::ostream& os, const RealCoord& rc){
  os << "x: " << rc.x << " y: " << rc.y;
  return os;
}

std::ostream& operator<<(std::ostream& os, const PixCoord& pc){
  os << "x: " << pc.x << " y: " << pc.y;
  return os;
}

float wiki_bilinear_interp2(const Eigen::Matrix3d& arr, const double& x, const double& y) {

  RealCoord p = RealCoord(x, y);

  // nearest neighbor real coordinates
  // top left (tl), top right (tr),
  // bottom left (bl), bottom right (br)
  auto [tl, tr, bl, br] = p.nearest_neighbors_coords();

  // weights per neighbor and normalization scale
  // which should technically always be 1
  double normalization = 1 / ((tr.x - tl.x) * (tl.y - bl.y));
  double w_tl = (tr.x - x) * (y - bl.y);
  double w_tr = (x - tl.x) * (y - bl.y);
  double w_bl = (br.x - x) * (tr.y - y);
  double w_br = (x - bl.x) * (tl.y - y);

  // array indices of nearest neighbor
  // top left (itl), top right (itr),
  // bottom left (ibl), bottom right (ibr)
  auto [itl, itr, ibl, ibr] = p.nearest_neighbors_idxs();

  return  normalization * (w_tl * arr(itl.x, itl.y) +
                          w_tr * arr(itr.x, itr.y) +
                          w_bl * arr(ibl.x, ibl.y) +
                          w_br * arr(ibr.x, ibr.y));

}


float origigi_bilinear_interp(const Eigen::Matrix3d& arr, const double x, const double y) {

  float ax = x + 0.5;
  float ay = y + 0.5;
  float a_px = floor(ax);
  float a_py = floor(ay);
  float a_amt = (ax - a_px) * (ay - a_py);

  // Bottom right
  float bx = x + 0.5;
  float by = y - 0.5;
  float b_px = floor(bx);
  float b_py = floor(by);
  float b_amt = (bx - b_px) * (b_py + 1.0 - by);

  // Bottom left
  float cx = x - 0.5;
  float cy = y - 0.5;
  float c_px = floor(cx);
  float c_py = floor(cy);
  float c_amt = (c_px + 1.0 - cx) * (c_py + 1.0 - cy);

  // Top left
  float dx = x - 0.5;
  float dy = y + 0.5;
  float d_px = floor(dx);
  float d_py = floor(dy);
  float d_amt = (d_px + 1.0 - dx) * (dy - d_py);

  return (a_amt * arr((int)a_px, (int)a_py) +
          b_amt * arr((int)b_px, (int)b_py) +
          c_amt * arr((int)c_px, (int)c_py) +
          d_amt * arr((int)d_px, (int)d_py));

}


float bilinear_interp(const Eigen::Matrix3d& arr, const double x, const double y) {

  // top left (tl), top right (tr),
  // bottom left (bl), bottom right (br)
  RealCoord rtl(x-0.5, y+0.5);
  RealCoord rtr(x+0.5, y+0.5);
  RealCoord rbl(x-0.5, y-0.5);
  RealCoord rbr(x+0.5, y-0.5);

  // pixel coordinates (quantized real coords)
  // top left (tl), top right (tr),
  // bottom left (bl), bottom right (br)
  PixCoord tl(rtl);
  PixCoord tr(rtr);
  PixCoord bl(rbl);
  PixCoord br(rbr);

  double normalization = 1;
  double w_tl = (tl.x + 1.0 - rtl.x) * (rtl.y - tl.y);
  double w_tr = (rtr.x - tr.x) * (rtr.y - tr.y);
  double w_bl = (bl.x + 1.0 - rbl.x) * (bl.y + 1.0 - rbl.y);
  double w_br = (rbr.x - br.x) * (br.y + 1.0 - rbr.y);


  return normalization * (w_tl * arr(tl.x, tl.y) +
                          w_tr * arr(tr.x, tr.y) +
                          w_bl * arr(bl.x, bl.y) +
                          w_br * arr(br.x, br.y));

}


Eigen::MatrixXd apply_mask(Eigen::Ref<Eigen::MatrixXd> img,
                   const Eigen::MatrixXi& mask,
                   int flags,
                   const std::vector<int>& exceptions) {

  int npixels = img.rows()*img.cols();

  // evaluated as lvalue because of auto
  auto mask_unravelled = mask.reshaped();
  auto array_unravelled = img.reshaped();

  for (int i=0; i<npixels; ++i){
    //int pix_flags = static_cast<int>(mask_unravelled(i));
    int pix_flags = mask_unravelled(i);
    bool is_exception = false;
    for (auto& e : exceptions){
      is_exception = is_exception || e == pix_flags;
      std::cout << "is exception " << is_exception << std::endl;
      if (!is_exception) std::cout << "    true! " << mask_unravelled(i) << " " << array_unravelled(i) <<
                          " " << (flags & pix_flags) <<  " " << (e == pix_flags) <<std::endl;
    }

    std::cout << "flags & pix " << " " << flags << " " << pix_flags << " " << (flags & pix_flags) << std::endl;
    if (!is_exception && ((flags & pix_flags) != 0)){ // Isn't this upside down?
      array_unravelled(i) = -9.0f;
    }
  }
  return img;
}


int main () {
  Eigen::Matrix2d kernel;
  kernel << 1, 2, 3, 4;
  std::cout << kernel << std::endl;

  Eigen::Matrix3d kernel2;
  kernel2 <<
    1, 0, 0,
    1, 0, 0,
    1, 0, 0;

  Eigen::Matrix3d kernel3;
  kernel3 <<
    1, 1, 0,
    1, 0, 0,
    1, 0, 0;

  // hah, everything is an expression, comma separator
  // special matrices. None of them actually exist in
  // memory unless forced to....
  Eigen::Matrix3d kernel4;
  (kernel4 <<
    1, 1, 0,
    1, 0, 0,
   1, 1, 0).finished();

  Eigen::Matrix3f kernel5;
  kernel5 <<
    1, 1, 1,
    1, 0, 1,
    1, 1, 1; //std::numeric_limits<double>::infinity();;

  // same like finished above, just forced to cast so constructor takes
  // care of it.
  int dim = 60;
  Eigen::MatrixXd kernel6 = Eigen::MatrixXd::Identity(dim, dim);
  Eigen::MatrixXi mask = Eigen::MatrixXi::Constant(dim, dim, 1);
  //std::cout << mask << std::endl;

  mask(0, 0) = 2;
  auto masked_arr = apply_mask(kernel6, mask, 2, {1,});
  std::cout << masked_arr << std::endl;


  float x = 1.1;
  float y = 1.1;

  //Eigen::Block<Eigen::Matrix2f> block = kernel5.block<2, 2>(1, 1);
  //block(0, 0) = 10.0f;
  Eigen::Matrix2f a = kernel5.block<2, 2>(1, 1);
  a(0, 0) = 10.0f;
  std::cout << kernel5 << std::endl;

  std::cout << "Elementwise operations of nan masked arrays: " << std::endl;
  std::cout << kernel5 << std::endl;
  std::cout << "max: " << kernel5.maxCoeff() << std::endl;
  std::cout << "min: " << kernel5.minCoeff() << std::endl;
  std::cout << "sum: " << kernel5.sum() << std::endl;



  //  std::cout << "Interpolation " << std::endl;
//  auto const t0 = std::chrono::high_resolution_clock::now();
//  for (auto i=0; i<100000000; i++)
//    origigi_bilinear_interp(kernel4, x, y);
//  auto const t1 = std::chrono::high_resolution_clock::now();
//  std::chrono::duration<double> durationb = t1 - t0;
//  std::cout << "Origigi duration: " << durationb.count() << "s\n";
//
//  auto const t0b = std::chrono::high_resolution_clock::now();
//  for (auto i=0; i<100000000; i++)
//    bilinear_interp(kernel4, x, y);
//  auto const t1b = std::chrono::high_resolution_clock::now();
//  std::chrono::duration<double> duration = t1b - t0b;
//  std::cout << "KBMOD duration: " << duration.count() << "s\n";
//
//  auto const t0a = std::chrono::high_resolution_clock::now();
//  for (auto i=0; i<100000000; i++)
//    wiki_bilinear_interp2(kernel4, x, y);
//  auto const t1a = std::chrono::high_resolution_clock::now();
//  std::chrono::duration<double> durationa = t1a - t0a;
//  std::cout << "WIKI duration: " << durationa.count() << "s\n";
//
//
//  std::cout << "Origigi implementation: " << std::endl;
//  std:: cout << bilinear_interp(kernel3, x, y) << std::endl;
//  std::cout << "KBMOD implementation: " << std::endl;
//  std:: cout << bilinear_interp(kernel3, x, y) << std::endl;
//  std::cout << "Wiki implementation: " << std::endl;
//  std:: cout << wiki_bilinear_interp2(kernel3, x, y) << std::endl;

}
