#ifndef MIXFORMER_H
#define MIXFORMER_H

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include "acl/acl.h"
#include "acl/acl_rt.h"

// -------------------------- 结构体定义 --------------------------
struct DrBBox {
    float x0;  // 左上角x
    float y0;  // 左上角y
    float x1;  // 右下角x
    float y1;  // 右下角y
};

struct DrOBB {
    DrBBox box;
    float score;
    int class_id;
};

struct Config {
    float template_factor = 2.0;
    float search_factor = 4.5;
    float template_size = 112;  // 模型输入img_t/img_ot尺寸
    float search_size = 224;    // 模型输入img_search尺寸
    float stride = 16;
    int feat_sz = 14;
    int update_interval = 200;  // 模板更新间隔
};

// -------------------------- 宏定义 --------------------------
#define DEVICE_ID 0
#define INPUT_NUM 3               // 输入张量数量
#define OUTPUT_NUM 2              // 输出张量数量
#define IMG_T_SIZE 1*3*112*112    // img_t: NCHW
#define IMG_OT_SIZE 1*3*112*112   // img_ot: NCHW
#define IMG_SEARCH_SIZE 1*3*224*224// img_search: NCHW
#define OUTPUT_BOX_SIZE 1*4       // pred_boxes: [cx, cy, w, h]
#define OUTPUT_SCORE_SIZE 1*1     // pred_scores: [confidence]

// 归一化参数（BGR→RGB后使用）
const float MEANS[3] = {0.406*255, 0.485*255, 0.456*255};
const float NORMS[3] = {1/(0.225*255), 1/(0.229*255), 1/(0.224*255)};

// -------------------------- Mixformer核心类 --------------------------
class Mixformer {
public:
    // 构造函数
    Mixformer(const std::string& model_path, aclrtContext context, aclrtStream stream);
    ~Mixformer();

    // 初始化（模型加载+缓冲区创建）
    int init();

    // 跟踪初始化（第一帧：设置初始目标框，提取模板）
    void init_track(const cv::Mat& img, DrOBB bbox);

    // 跟踪主函数（后续帧：提取搜索区域，推理，更新目标框）
    const DrOBB& track(const cv::Mat& img);

    // 释放资源
    int release();

private:
    // ACL错误检查）
    int check_acl_error(aclError err, const std::string& step, const std::string& msg) const;

    // 预处理：图像裁剪+Resize+归一化+BGR2RGB+NCHW转换
    int preprocess(const cv::Mat& z_patch, const cv::Mat& oz_patch, const cv::Mat& x_patch);

    // 推理
    int infer();

    // 后处理：解析输出张量，映射坐标到原图
    // int postprocess(float resize_factor, int img_width, int img_height);
    int postprocess(const cv::Mat& img, float resize_factor, int img_width, int img_height);

    // 提取目标区域
    void sample_target(const cv::Mat& im, cv::Mat& croped, DrBBox target_bb, 
                      float search_area_factor, int output_sz, float& resize_factor);

    // 归一化（inplace）
    void normalize_inplace(cv::Mat& mat_inplace, const float mean[3], const float scale[3]);

    // 坐标映射：将模型输出映射到原图
    void map_box_back(DrBBox& pred_box, float resize_factor);

    // 边界裁剪：确保目标框在图像内
    void clip_box(DrBBox& box, int height, int width, int margin);

    aclrtContext context_ = nullptr;  // ACL上下文
    aclrtStream stream_ = nullptr;    // ACL流
    aclrtRunMode run_mode_ = ACL_HOST; // 运行模式（HOST/DEVICE）

    // 模型资源
    std::string model_path_;          // 模型路径
    uint32_t model_id_ = 0;           // 模型ID
    aclmdlDesc* model_desc_ = nullptr;// 模型描述符
    size_t model_mem_size_ = 0;       // 模型工作内存大小
    size_t model_weight_size_ = 0;    // 模型权重内存大小
    void* model_mem_ptr_ = nullptr;   // 模型工作内存指针
    void* model_weight_ptr_ = nullptr;// 模型权重内存指针

    // 输入输出缓冲区（DEVICE端）
    float* dev_img_t_ = nullptr;      // 输入0：img_t（模板）
    float* dev_img_ot_ = nullptr;     // 输入1：img_ot（在线模板）
    float* dev_img_search_ = nullptr; // 输入2：img_search（搜索区域）
    float* dev_output_box_ = nullptr; // 输出0：pred_boxes
    float* dev_output_score_ = nullptr;// 输出1：pred_scores

    // 输入输出缓冲区（HOST端，用于数据预处理/后处理）
    float* host_img_t_ = nullptr;
    float* host_img_ot_ = nullptr;
    float* host_img_search_ = nullptr;
    float* host_output_box_ = nullptr;
    float* host_output_score_ = nullptr;

    // 缓冲区大小
    size_t input0_size_ = IMG_T_SIZE * sizeof(float);
    size_t input1_size_ = IMG_OT_SIZE * sizeof(float);
    size_t input2_size_ = IMG_SEARCH_SIZE * sizeof(float);
    size_t output0_size_ = OUTPUT_BOX_SIZE * sizeof(float);
    size_t output1_size_ = OUTPUT_SCORE_SIZE * sizeof(float);

    // 跟踪状态变量
    bool is_inited_ = false;          // 模型初始化标记
    bool is_track_inited_ = false;    // 跟踪初始化标记
    DrBBox state_;                    // 当前跟踪目标框
    Config cfg_;                      // 跟踪配置参数
    cv::Mat z_patch_;                 // 初始模板
    cv::Mat oz_patch_;                // 在线更新模板
    cv::Mat max_oz_patch_;            // 最优模板
    DrOBB object_box_;                // 输出目标框
    int frame_id_ = 0;                // 帧计数器
    float max_pred_score_ = -1.f;     // 最大预测分数
    float max_score_decay_ = 1.f;     // 分数衰减系数
};

#endif // MIXFORMER_H