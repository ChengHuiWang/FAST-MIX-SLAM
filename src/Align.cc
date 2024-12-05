#include "Align.h"

using namespace Eigen;

namespace mix {

      	double ResponseInverse[256] = { 0.00841609, 0.0117425, 0.0163715, 0.0227062, 0.0307793, 0.0420908, 0.0563273, 0.0702391, 0.081858, 0.0902191, 0.0969666, 0.103887, 0.111077, 0.1186, 0.126433, 0.134387, 0.142256, 0.149642, 0.156204, 0.161906, 0.167166, 0.172786, 0.179228, 0.186216, 0.193467, 0.200874, 0.208582, 0.216773, 0.225713, 0.234566, 0.242634, 0.250073, 0.256985, 0.263367, 0.269571, 0.275992, 0.282901, 0.290243, 0.297663, 0.305391, 0.31357, 0.321669, 0.329327, 0.336768, 0.344082, 0.351355, 0.358622, 0.36618, 0.374049, 0.382255, 0.390415, 0.398601, 0.406658, 0.414845, 0.423155, 0.431489, 0.439401, 0.447226, 0.455077, 0.463002, 0.470814, 0.478383, 0.485407, 0.491809, 0.497933, 0.504507, 0.511623, 0.5189, 0.52608, 0.533122, 0.540422, 0.54801, 0.555832, 0.564024, 0.572326, 0.580513, 0.588727, 0.597008, 0.605427, 0.613834, 0.621943, 0.629932, 0.637915, 0.645967, 0.653575, 0.661065, 0.668539, 0.676072, 0.683802, 0.691179, 0.698171, 0.704728, 0.710978, 0.717589, 0.724899, 0.733345, 0.743074, 0.753592, 0.764423, 0.774916, 0.784384, 0.792826, 0.800573, 0.808106, 0.815808, 0.823578, 0.831077, 0.838145, 0.844756, 0.851041, 0.856816, 0.861678, 0.866557, 0.871526, 0.877287, 0.883689, 0.890928, 0.899015, 0.907877, 0.917444, 0.927336, 0.937155, 0.946747, 0.956209, 0.965703, 0.975399, 0.984787, 0.99401, 1.00274, 1.01078, 1.01799, 1.02442, 1.03046, 1.03619, 1.04179, 1.04796, 1.05506, 1.06315, 1.07207, 1.0815, 1.09109, 1.1005, 1.1094, 1.11761, 1.12522, 1.13256, 1.13996, 1.14729, 1.15454, 1.16158, 1.16827, 1.17474, 1.18114, 1.18763, 1.19451, 1.20198, 1.21009, 1.21853, 1.22692, 1.23553, 1.24383, 1.2513, 1.2583, 1.26541, 1.27342, 1.2825, 1.29273, 1.30334, 1.31355, 1.32308, 1.33116, 1.33847, 1.34534, 1.35162, 1.35714, 1.36176, 1.36559, 1.36979, 1.37509, 1.38203, 1.39075, 1.4014, 1.41422, 1.42784, 1.44088, 1.45246, 1.46233, 1.47059, 1.47682, 1.48165, 1.48558, 1.48914, 1.493, 1.49704, 1.50163, 1.50692, 1.51283, 1.51947, 1.52713, 1.5357, 1.54509, 1.5548, 1.56454, 1.57418, 1.58342, 1.59201, 1.59955, 1.60669, 1.61394, 1.62142, 1.62908, 1.63679, 1.64437, 1.65204, 1.66015, 1.66841, 1.67663, 1.68477, 1.69215, 1.69909, 1.70621, 1.71386, 1.722, 1.7307, 1.74008, 1.74976, 1.75899, 1.768, 1.77664, 1.78581, 1.79608, 1.8064, 1.81622, 1.8249, 1.83224, 1.83846, 1.84392, 1.84864, 1.85313, 1.85804, 1.86336, 1.86959, 1.87686, 1.88501, 1.89281, 1.89978, 1.90533, 1.91059, 1.91702, 1.92686, 1.94244, 1.96731, 2.00689, 2.06997, 2.16496, 2.30425 };

// SSE 就交给你们了
    bool Align2D(
            const cv::Mat &cur_img,
            uint8_t *ref_patch_with_border,
            uint8_t *ref_patch,
            const int n_iter,
            Vector2f &cur_px_estimate,
	   float *ExposureTime_A ,            //曝光时间
	   int ReferForAlignID ,
           int nimg_A  ,  
            bool no_simd
	) {
        const int halfpatch_size_ = 4;
        const int patch_size_ = 8;
        const int patch_area_ = 64;
        bool converged = false;

        // compute derivative of template and prepare inverse compositional
        float __attribute__ (( __aligned__ ( 16 ))) ref_patch_dx[patch_area_];
        float __attribute__ (( __aligned__ ( 16 ))) ref_patch_dy[patch_area_];
        Matrix3f H;
        H.setZero();

        // compute gradient and hessian
        const int ref_step = patch_size_ + 2;
        float *it_dx = ref_patch_dx;
        float *it_dy = ref_patch_dy;
        for (int y = 0; y < patch_size_; ++y) {
            uint8_t *it = ref_patch_with_border + (y + 1) * ref_step + 1;
            for (int x = 0; x < patch_size_; ++x, ++it, ++it_dx, ++it_dy) {
                Vector3f J;
                J[0] = 0.5 * (it[1] - it[-1]);
                J[1] = 0.5 * (it[ref_step] - it[-ref_step]);
                J[2] = 1;
                *it_dx = J[0];
                *it_dy = J[1];
                H += J * J.transpose();
            }
        }
        Matrix3f Hinv = H.inverse();
        float mean_diff = 0;

        // Compute pixel location in new image:
        float u = cur_px_estimate.x();
        float v = cur_px_estimate.y();

        // termination condition
        const float min_update_squared = 0.03 * 0.03;
        const int cur_step = cur_img.step.p[0];
        Vector3f update;
        update.setZero();
        float chi2 = 0;
        for (int iter = 0; iter < n_iter; ++iter) {
            chi2 = 0;
            int u_r = floor(u);
            int v_r = floor(v);
            if (u_r < halfpatch_size_ || v_r < halfpatch_size_ || u_r >= cur_img.cols - halfpatch_size_ ||
                v_r >= cur_img.rows - halfpatch_size_)
                break;

            if (isnan(u) ||
                isnan(v)) // TODO very rarely this can happen, maybe H is singular? should not be at corner.. check
                return false;

            // compute interpolation weights
            float subpix_x = u - u_r;
            float subpix_y = v - v_r;
            float wTL = (1.0 - subpix_x) * (1.0 - subpix_y);
            float wTR = subpix_x * (1.0 - subpix_y);
            float wBL = (1.0 - subpix_x) * subpix_y;
            float wBR = subpix_x * subpix_y;

            // loop through search_patch, interpolate
            uint8_t *it_ref = ref_patch;
            float *it_ref_dx = ref_patch_dx;
            float *it_ref_dy = ref_patch_dy;
            Vector3f Jres;
            Jres.setZero();
            for (int y = 0; y < patch_size_; ++y) {
                uint8_t *it = (uint8_t *) cur_img.data + (v_r + y - halfpatch_size_) * cur_step + u_r - halfpatch_size_;
                for (int x = 0; x < patch_size_; ++x, ++it, ++it_ref, ++it_ref_dx, ++it_ref_dy) {
                    float search_pixel = wTL * it[0] + wTR * it[1] + wBL * it[cur_step] + wBR * it[cur_step + 1];
                    float res = search_pixel - *it_ref + mean_diff;
		  //  float res = search_pixel - ResponseFunction_A(ExposureTime_A[nimg_A]*ResponseInverse[ int(*it_ref )] /ExposureTime_A[ReferForAlignID]) + mean_diff;
		// cout<<"ResponseFunction_A="<<ResponseFunction_A(ExposureTime_A[nimg_A]*ResponseInverse[ int(*it_ref )] /ExposureTime_A[ReferForAlignID])<<endl;
		//    cout<<"search_pixel="<<search_pixel<<";  *it_ref = "<<float(*it_ref)<<endl;
	      //    cout<<"pKF->mnFrameId="<<ReferForAlignID<<"  currentFrameId="<<nimg_A<<endl;
		 //    cout<<"ExposureTime_A="<<ExposureTime_A[nimg_A]<<";  ReferForAlignID="<<ReferForAlignID<<endl;
                    Jres[0] -= res * (*it_ref_dx);
                    Jres[1] -= res * (*it_ref_dy);
                    Jres[2] -= res;
                    chi2 += res * res;
                }
            }
            update = Hinv * Jres;
            u += update[0];
            v += update[1];
            mean_diff += update[2];
            if (update[0] * update[0] + update[1] * update[1] < min_update_squared) {
                converged = true;
                break;
            }
        }

        cur_px_estimate << u, v;
        return converged;
    }
    
 /*    void  getExposureTime_A(float *T)
    {
        ExposureTime_A=T;
    }

    void  getNImage_A(int N )
   {
        nimg_A=N;
   }
    
    void  getReferForAlignID(int N )
   {
        ReferForAlignID=N;
   }*/
   
   double ResponseFunction_A(double x)
{

	//多项式拟合
        double a1 = 256.3 ;
	double b1 = 2.164 ;
	double c1 = 0.9676;
	double a2 = 65.64;
	double b2 = 1.026;
	double c2 = 0.5621;
	double a3 = 24.68;
	double b3 = 0.4968;
	double c3 = 0.3377;
	double y = a1*exp(-pow(((x - b1) / c1), 2)) + a2*exp(-pow(((x - b2) / c2), 2)) + a3*exp(-pow(((x - b3) / c3), 2));
	
	return y;

}

}
