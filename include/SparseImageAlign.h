#ifndef MIX_SPARSE_IMAGE_ALIGN_
#define MIX_SPARSE_IMAGE_ALIGN_

#include "Common.h"
#include "NLSSolver.h"
#include "Frame.h"

//float ExposureTime[2000];              //采用 string 类型，存100行的文本，不要用数组 
//int img_n = 0;

// 稀疏直接法求解器
// 请注意SVO的直接法用了一种逆向的奇怪解法，它的雅可比是在Ref中估计而不是在Current里估计的，所以迭代过程中雅可比是不动的

namespace mix {

    /// Optimize the pose of the frame by minimizing the photometric error of feature patches.
    class SparseImgAlign : public NLLSSolver<6, SE3f> {
        static const int patch_halfsize_ = 2;
        static const int patch_size_ = 2 * patch_halfsize_;
        static const int patch_area_ = patch_size_ * patch_size_;
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        cv::Mat resimg_;

        /**
         * @brief constructor
         * @param[in] n_levels total pyramid level
         * @param[in] min_level minimum levels
         * @param[in] n_iter iterations
         * @param[in] methos GaussNewton or LevernbergMarquardt
         * @param[in] display display the residual image
         * @param[in] verbose output the inner computation information
         */
        SparseImgAlign(
                int n_levels,
                int min_level,
                int n_iter = 10,
                Method method = GaussNewton,
                bool display = false,
                bool verbose = false);

        /**
         * 计算 ref 和 current 之间的运动
         * @brief compute the relative motion between ref frame and current frame
         * @param[in] ref_frame the reference
         * @param[in] cur_frame the current frame
         * @param[out] TCR motion from ref to current
         */
        size_t run(
                Frame *ref_frame,
                Frame *cur_frame,
                SE3f &TCR
        );

        /// Return fisher information matrix, i.e. the Hessian of the log-likelihood
        /// at the converged state.
        Matrix<float, 6, 6> getFisherInformation();

	///////////////////////////自定义////////////////////////////
	float *ExposureT;              //Sparse曝光时间
        int nimgs = 0;
	void getExposureTime(float *);
	void getNImages(int ); 
	double ResponseInverse[256] = { 0.00841609, 0.0117425, 0.0163715, 0.0227062, 0.0307793, 0.0420908, 0.0563273, 0.0702391, 0.081858, 0.0902191, 0.0969666, 0.103887, 0.111077, 0.1186, 0.126433, 0.134387, 0.142256, 0.149642, 0.156204, 0.161906, 0.167166, 0.172786, 0.179228, 0.186216, 0.193467, 0.200874, 0.208582, 0.216773, 0.225713, 0.234566, 0.242634, 0.250073, 0.256985, 0.263367, 0.269571, 0.275992, 0.282901, 0.290243, 0.297663, 0.305391, 0.31357, 0.321669, 0.329327, 0.336768, 0.344082, 0.351355, 0.358622, 0.36618, 0.374049, 0.382255, 0.390415, 0.398601, 0.406658, 0.414845, 0.423155, 0.431489, 0.439401, 0.447226, 0.455077, 0.463002, 0.470814, 0.478383, 0.485407, 0.491809, 0.497933, 0.504507, 0.511623, 0.5189, 0.52608, 0.533122, 0.540422, 0.54801, 0.555832, 0.564024, 0.572326, 0.580513, 0.588727, 0.597008, 0.605427, 0.613834, 0.621943, 0.629932, 0.637915, 0.645967, 0.653575, 0.661065, 0.668539, 0.676072, 0.683802, 0.691179, 0.698171, 0.704728, 0.710978, 0.717589, 0.724899, 0.733345, 0.743074, 0.753592, 0.764423, 0.774916, 0.784384, 0.792826, 0.800573, 0.808106, 0.815808, 0.823578, 0.831077, 0.838145, 0.844756, 0.851041, 0.856816, 0.861678, 0.866557, 0.871526, 0.877287, 0.883689, 0.890928, 0.899015, 0.907877, 0.917444, 0.927336, 0.937155, 0.946747, 0.956209, 0.965703, 0.975399, 0.984787, 0.99401, 1.00274, 1.01078, 1.01799, 1.02442, 1.03046, 1.03619, 1.04179, 1.04796, 1.05506, 1.06315, 1.07207, 1.0815, 1.09109, 1.1005, 1.1094, 1.11761, 1.12522, 1.13256, 1.13996, 1.14729, 1.15454, 1.16158, 1.16827, 1.17474, 1.18114, 1.18763, 1.19451, 1.20198, 1.21009, 1.21853, 1.22692, 1.23553, 1.24383, 1.2513, 1.2583, 1.26541, 1.27342, 1.2825, 1.29273, 1.30334, 1.31355, 1.32308, 1.33116, 1.33847, 1.34534, 1.35162, 1.35714, 1.36176, 1.36559, 1.36979, 1.37509, 1.38203, 1.39075, 1.4014, 1.41422, 1.42784, 1.44088, 1.45246, 1.46233, 1.47059, 1.47682, 1.48165, 1.48558, 1.48914, 1.493, 1.49704, 1.50163, 1.50692, 1.51283, 1.51947, 1.52713, 1.5357, 1.54509, 1.5548, 1.56454, 1.57418, 1.58342, 1.59201, 1.59955, 1.60669, 1.61394, 1.62142, 1.62908, 1.63679, 1.64437, 1.65204, 1.66015, 1.66841, 1.67663, 1.68477, 1.69215, 1.69909, 1.70621, 1.71386, 1.722, 1.7307, 1.74008, 1.74976, 1.75899, 1.768, 1.77664, 1.78581, 1.79608, 1.8064, 1.81622, 1.8249, 1.83224, 1.83846, 1.84392, 1.84864, 1.85313, 1.85804, 1.86336, 1.86959, 1.87686, 1.88501, 1.89281, 1.89978, 1.90533, 1.91059, 1.91702, 1.92686, 1.94244, 1.96731, 2.00689, 2.06997, 2.16496, 2.30425 };
	///////////////////////////////////////////////////////////////////
	
    protected:
        Frame *ref_frame_;              //!< reference frame, has depth for gradient pixels.
        Frame *cur_frame_;              //!< only the image is known!
        int level_;                     //!< current pyramid level on which the optimization runs.
        bool display_;                  //!< display residual image.
        int max_level_;                 //!< coarsest pyramid level for the alignment.
        int min_level_;                 //!< finest pyramid level for the alignment.

        // cache:
        Matrix<float, 6, Dynamic, ColMajor> jacobian_cache_;    // 雅可比矩阵，这个东西是固定下来的

        bool have_ref_patch_cache_;
        cv::Mat ref_patch_cache_;
        std::vector<bool> visible_fts_;

        // 在ref中计算雅可比
        void precomputeReferencePatches();

        // 派生出来的虚函数
        virtual float computeResiduals(const SE3f &model, bool linearize_system, bool compute_weight_scale = false);

        virtual int solve();

        virtual void update(const ModelType &old_model, ModelType &new_model);

        virtual void startIteration();

        virtual void finishIteration();
	
	//////////////////////////////wch//////////////////////////////
	 virtual   double Response_Inverse(double x);
	 
	 virtual   double  ResponseFunction(double x);
	 ////////////////////////////////////////////////////////////////

        // *************************************************************************************
        // 一些固定的雅可比
        // xyz 到 相机坐标 的雅可比，平移在前
        // 这里已经取了负号，不要再取一遍！
        inline Eigen::Matrix<float, 2, 6> JacobXYZ2Cam(const Vector3f &xyz) {
            Eigen::Matrix<float, 2, 6> J;
            const float x = xyz[0];
            const float y = xyz[1];
            const float z_inv = 1. / xyz[2];
            const float z_inv_2 = z_inv * z_inv;

            J(0, 0) = -z_inv;           // -1/z
            J(0, 1) = 0.0;              // 0
            J(0, 2) = x * z_inv_2;        // x/z^2
            J(0, 3) = y * J(0, 2);      // x*y/z^2
            J(0, 4) = -(1.0 + x * J(0, 2)); // -(1.0 + x^2/z^2)
            J(0, 5) = y * z_inv;          // y/z

            J(1, 0) = 0.0;              // 0
            J(1, 1) = -z_inv;           // -1/z
            J(1, 2) = y * z_inv_2;        // y/z^2
            J(1, 3) = 1.0 + y * J(1, 2); // 1.0 + y^2/z^2
            J(1, 4) = -J(0, 3);       // -x*y/z^2
            J(1, 5) = -x * z_inv;         // x/z
            return J;
        }

    };

}// namespace mix 


#endif