#include "mixformer.h"
#include <iostream>
#include <cstring>
#include <chrono>
#include <errno.h>

using namespace std;
using namespace cv;

// -------------------------- 构造函数 --------------------------
Mixformer::Mixformer(const std::string& model_path, aclrtContext context, aclrtStream stream)
    : model_path_(model_path), context_(context), stream_(stream) {
    // 初始化HOST端缓冲区
    host_img_t_ = new float[IMG_T_SIZE];
    host_img_ot_ = new float[IMG_OT_SIZE];
    host_img_search_ = new float[IMG_SEARCH_SIZE];
    host_output_box_ = new float[OUTPUT_BOX_SIZE];
    host_output_score_ = new float[OUTPUT_SCORE_SIZE];

    memset(host_img_t_, 0, IMG_T_SIZE * sizeof(float));
    memset(host_img_ot_, 0, IMG_OT_SIZE * sizeof(float));
    memset(host_img_search_, 0, IMG_SEARCH_SIZE * sizeof(float));
    memset(host_output_box_, 0, OUTPUT_BOX_SIZE * sizeof(float));
    memset(host_output_score_, 0, OUTPUT_SCORE_SIZE * sizeof(float));

    cout << "[Mixformer] 初始化: 模型路径=" << model_path_ << endl;
    cout << "[Mixformer] 输入缓冲区大小: img_t=" << input0_size_/1024 << "KB, img_ot=" << input1_size_/1024 << "KB, img_search=" << input2_size_/1024 << "KB" << endl;
    cout << "[Mixformer] 输出缓冲区大小: pred_boxes=" << output0_size_ << "字节, pred_scores=" << output1_size_ << "字节" << endl;
}

// -------------------------- 析构函数 --------------------------
Mixformer::~Mixformer() {
    release();
    // 释放HOST端缓冲区
    delete[] host_img_t_;
    delete[] host_img_ot_;
    delete[] host_img_search_;
    delete[] host_output_box_;
    delete[] host_output_score_;
}

// -------------------------- ACL错误检查 --------------------------
int Mixformer::check_acl_error(aclError err, const std::string& step, const std::string& msg) const {
    if (err != ACL_SUCCESS) {
        cerr << "[ERROR] 步骤[" << step << "] " << msg << " 失败! 错误码:" << err << endl;
        return -1;
    }
    cout << "[INFO] 步骤[" << step << "] " << msg << " 成功!" << endl;
    return 0;
}

// -------------------------- 模型初始化（加载+设备端缓冲区创建） --------------------------
int Mixformer::init() {
    if (is_inited_) {
        cout << "[INFO] Mixformer已初始化，无需重复操作" << endl;
        return 0;
    }

    aclError err;

    // 1. 获取运行模式（HOST/DEVICE）
    err = aclrtGetRunMode(&run_mode_);
    if (check_acl_error(err, "1", "获取运行模式") != 0) return -1;

    // 2. 查询模型内存大小
    err = aclmdlQuerySize(model_path_.c_str(), &model_mem_size_, &model_weight_size_);
    if (check_acl_error(err, "2", "查询模型内存大小") != 0) return -1;
    cout << "[Mixformer] 模型工作内存大小: " << model_mem_size_/1024/1024 << "MB, 权重内存大小: " << model_weight_size_/1024/1024 << "MB" << endl;

    // 3. 分配模型显存
    err = aclrtMalloc(&model_mem_ptr_, model_mem_size_, ACL_MEM_MALLOC_HUGE_FIRST);
    if (check_acl_error(err, "3", "分配模型工作内存") != 0) return -1;
    
    err = aclrtMalloc(&model_weight_ptr_, model_weight_size_, ACL_MEM_MALLOC_HUGE_FIRST);
    if (check_acl_error(err, "4", "分配模型权重内存") != 0) {
        aclrtFree(model_mem_ptr_);
        return -1;
    }

    // 4. 加载模型到NPU
    err = aclmdlLoadFromFileWithMem(
        model_path_.c_str(), &model_id_,
        model_mem_ptr_, model_mem_size_,
        model_weight_ptr_, model_weight_size_
    );
    if (check_acl_error(err, "5", "加载模型到NPU") != 0) {
        aclrtFree(model_weight_ptr_);
        aclrtFree(model_mem_ptr_);
        return -1;
    }

    // 5. 创建模型描述符
    model_desc_ = aclmdlCreateDesc();
    err = aclmdlGetDesc(model_desc_, model_id_);
    if (check_acl_error(err, "6", "获取模型描述符") != 0) {
        aclmdlUnload(model_id_);
        aclrtFree(model_weight_ptr_);
        aclrtFree(model_mem_ptr_);
        return -1;
    }

    // 6. 分配设备端输入输出缓冲区
    err = aclrtMalloc((void**)&dev_img_t_, input0_size_, ACL_MEM_MALLOC_HUGE_FIRST);
    if (check_acl_error(err, "7", "分配img_t设备缓冲区") != 0) return -1;
    
    err = aclrtMalloc((void**)&dev_img_ot_, input1_size_, ACL_MEM_MALLOC_HUGE_FIRST);
    if (check_acl_error(err, "8", "分配img_ot设备缓冲区") != 0) return -1;
    
    err = aclrtMalloc((void**)&dev_img_search_, input2_size_, ACL_MEM_MALLOC_HUGE_FIRST);
    if (check_acl_error(err, "9", "分配img_search设备缓冲区") != 0) return -1;
    
    err = aclrtMalloc((void**)&dev_output_box_, output0_size_, ACL_MEM_MALLOC_HUGE_FIRST);
    if (check_acl_error(err, "10", "分配pred_boxes设备缓冲区") != 0) return -1;
    
    err = aclrtMalloc((void**)&dev_output_score_, output1_size_, ACL_MEM_MALLOC_HUGE_FIRST);
    if (check_acl_error(err, "11", "分配pred_scores设备缓冲区") != 0) return -1;

    is_inited_ = true;
    cout << "[INFO] Mixformer初始化完成!" << endl;
    return 0;
}

// -------------------------- 跟踪初始化（第一帧） --------------------------
void Mixformer::init_track(const cv::Mat& img, DrOBB bbox) {
    if (!is_inited_) {
        cerr << "[ERROR] 模型未初始化，请先调用init()" << endl;
        return;
    }

    // 提取初始模板（img_t/img_ot）
    float resize_factor = 1.f;
    sample_target(img, z_patch_, bbox.box, cfg_.template_factor, cfg_.template_size, resize_factor);
    oz_patch_ = z_patch_.clone();
    max_oz_patch_ = z_patch_.clone();
    state_ = bbox.box;
    object_box_ = bbox;
    is_track_inited_ = true;
    frame_id_ = 0;
    max_pred_score_ = -1.f;

    cout << "[Mixformer] 跟踪初始化完成: 初始目标框=(" << bbox.box.x0 << "," << bbox.box.y0 << "," << bbox.box.x1 << "," << bbox.box.y1 << "), 置信度=" << bbox.score << endl;
}

// -------------------------- 归一化（inplace） --------------------------
void Mixformer::normalize_inplace(cv::Mat& mat_inplace, const float mean[3], const float scale[3]) {
    if (mat_inplace.type() != CV_32FC3) mat_inplace.convertTo(mat_inplace, CV_32FC3);
    for (int i = 0; i < mat_inplace.rows; ++i) {
        Vec3f* p = mat_inplace.ptr<Vec3f>(i);
        for (int j = 0; j < mat_inplace.cols; ++j) {
            p[j][0] = (p[j][0] - mean[0]) * scale[0];
            p[j][1] = (p[j][1] - mean[1]) * scale[1];
            p[j][2] = (p[j][2] - mean[2]) * scale[2];
        }
    }
}

// -------------------------- 提取目标区域（裁剪+Resize+填充） --------------------------
void Mixformer::sample_target(const cv::Mat& im, cv::Mat& croped, DrBBox target_bb, 
                             float search_area_factor, int output_sz, float& resize_factor) {
    // 检查输入图像是否为空
    if (im.empty()) {
        cerr << "[WARNING] sample_target: 输入图像为空，返回默认裁剪区域" << endl;
        croped = Mat::zeros(output_sz, output_sz, CV_8UC3);
        resize_factor = 1.0f;
        return;
    }

    int img_h = im.rows;
    int img_w = im.cols;

    // 目标框坐标
    float x0 = std::max(0.f, target_bb.x0);
    float y0 = std::max(0.f, target_bb.y0);
    float x1 = std::min((float)img_w, target_bb.x1);
    float y1 = std::min((float)img_h, target_bb.y1);

    // 目标框宽高
    float w = x1 - x0;
    float h = y1 - y0;
    if (w <= 0 || h <= 0) {
        cerr << "[WARNING] sample_target: 目标框宽高为0，返回默认裁剪区域" << endl;
        croped = Mat::zeros(output_sz, output_sz, CV_8UC3);
        resize_factor = 1.0f;
        return;
    }

    // 计算裁剪区域大小（基于目标框+搜索因子）
    float crop_sz = std::ceil(std::sqrt(w * h) * search_area_factor);
    if (crop_sz <= 0) crop_sz = output_sz;  // 避免裁剪尺寸为0

    // 裁剪区域中心（基于目标框中心）
    float cx = x0 + 0.5 * w;
    float cy = y0 + 0.5 * h;

    // 计算裁剪区域的左上角和右下角
    int x1_crop = std::round(cx - crop_sz * 0.5);
    int y1_crop = std::round(cy - crop_sz * 0.5);
    int x2_crop = x1_crop + crop_sz;
    int y2_crop = y1_crop + crop_sz;

    // 计算填充量（处理边界溢出）
    int left_pad = std::max(0, -x1_crop);
    int right_pad = std::max(0, x2_crop - img_w);
    int top_pad = std::max(0, -y1_crop);
    int bottom_pad = std::max(0, y2_crop - img_h);

    // 调整裁剪区域到图像边界内
    x1_crop += left_pad;
    y1_crop += top_pad;
    x2_crop = x1_crop + (x2_crop - x1_crop) - right_pad;
    y2_crop = y1_crop + (y2_crop - y1_crop) - bottom_pad;

    // 确保裁剪区域有效
    if (x1_crop >= x2_crop || y1_crop >= y2_crop) {
        cerr << "[WARNING] sample_target: 裁剪区域无效，返回默认裁剪区域" << endl;
        croped = Mat::zeros(output_sz, output_sz, CV_8UC3);
        resize_factor = 1.0f;
        return;
    }

    // 裁剪ROI
    cv::Rect roi_rect(x1_crop, y1_crop, x2_crop - x1_crop, y2_crop - y1_crop);
    cv::Mat roi = im(roi_rect);

    // 填充边界（如果需要）
    if (left_pad > 0 || right_pad > 0 || top_pad > 0 || bottom_pad > 0) {
        cv::copyMakeBorder(roi, croped, top_pad, bottom_pad, left_pad, right_pad, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    } else {
        croped = roi.clone();
    }

    // Resize到模型输入尺寸
    cv::resize(croped, croped, cv::Size(output_sz, output_sz));

    // 计算缩放因子（输出尺寸 / 原始裁剪尺寸）
    resize_factor = output_sz / crop_sz;

    cout << "[Mixformer] sample_target: 裁剪区域=(" << x1_crop << "," << y1_crop << "," << x2_crop << "," << y2_crop 
         << "), 填充量=(" << top_pad << "," << bottom_pad << "," << left_pad << "," << right_pad 
         << "), 缩放因子=" << resize_factor << endl;
}

// -------------------------- 预处理（图像→模型输入格式） --------------------------
int Mixformer::preprocess(const cv::Mat& z_patch, const cv::Mat& oz_patch, const cv::Mat& x_patch) {
    // 1. BGR→RGB转换
    Mat z_rgb, oz_rgb, x_rgb;
    cvtColor(z_patch, z_rgb, COLOR_BGR2RGB);
    cvtColor(oz_patch, oz_rgb, COLOR_BGR2RGB);
    cvtColor(x_patch, x_rgb, COLOR_BGR2RGB);

    // 2. 归一化（U8→FLOAT32，减均值除方差）
    normalize_inplace(z_rgb, MEANS, NORMS);
    normalize_inplace(oz_rgb, MEANS, NORMS);
    normalize_inplace(x_rgb, MEANS, NORMS);

    // 3. HWC→NCHW转换（OpenCV→模型输入格式）
    int ch = 3, h = cfg_.template_size, w = cfg_.template_size;
    int search_h = cfg_.search_size, search_w = cfg_.search_size;

    // img_t: [1,3,112,112]
    for (int c = 0; c < ch; ++c) {
        for (int i = 0; i < h; ++i) {
            for (int j = 0; j < w; ++j) {
                int idx = c * h * w + i * w + j;
                host_img_t_[idx] = z_rgb.ptr<Vec3f>(i)[j][c];
            }
        }
    }

    // img_ot: [1,3,112,112]
    for (int c = 0; c < ch; ++c) {
        for (int i = 0; i < h; ++i) {
            for (int j = 0; j < w; ++j) {
                int idx = c * h * w + i * w + j;
                host_img_ot_[idx] = oz_rgb.ptr<Vec3f>(i)[j][c];
            }
        }
    }

    // img_search: [1,3,224,224]
    for (int c = 0; c < ch; ++c) {
        for (int i = 0; i < search_h; ++i) {
            for (int j = 0; j < search_w; ++j) {
                int idx = c * search_h * search_w + i * search_w + j;
                host_img_search_[idx] = x_rgb.ptr<Vec3f>(i)[j][c];
            }
        }
    }

    cout << "[Mixformer] 预处理完成: HWC→NCHW, 归一化完成" << endl;
    return 0;
}

// -------------------------- 推理（ACL模型执行） --------------------------
int Mixformer::infer() {
    aclError err;
    aclmdlDataset* input_dataset = nullptr;
    aclmdlDataset* output_dataset = nullptr;
    aclDataBuffer* input_buf0 = nullptr;
    aclDataBuffer* input_buf1 = nullptr;
    aclDataBuffer* input_buf2 = nullptr;
    aclDataBuffer* output_buf0 = nullptr;
    aclDataBuffer* output_buf1 = nullptr;

    // 1. 创建输入数据集
    input_dataset = aclmdlCreateDataset();
    if (input_dataset == nullptr) throw runtime_error("创建输入数据集失败");

    // 1.1 添加img_t输入缓冲区
    input_buf0 = aclCreateDataBuffer(dev_img_t_, input0_size_);
    err = aclmdlAddDatasetBuffer(input_dataset, input_buf0);
    if (check_acl_error(err, "infer-1", "添加img_t输入缓冲区") != 0) throw runtime_error("添加img_t失败");

    // 1.2 添加img_ot输入缓冲区
    input_buf1 = aclCreateDataBuffer(dev_img_ot_, input1_size_);
    err = aclmdlAddDatasetBuffer(input_dataset, input_buf1);
    if (check_acl_error(err, "infer-2", "添加img_ot输入缓冲区") != 0) throw runtime_error("添加img_ot失败");

    // 1.3 添加img_search输入缓冲区
    input_buf2 = aclCreateDataBuffer(dev_img_search_, input2_size_);
    err = aclmdlAddDatasetBuffer(input_dataset, input_buf2);
    if (check_acl_error(err, "infer-3", "添加img_search输入缓冲区") != 0) throw runtime_error("添加img_search失败");

    // 2. 创建输出数据集
    output_dataset = aclmdlCreateDataset();
    if (output_dataset == nullptr) throw runtime_error("创建输出数据集失败");

    // 2.1 添加pred_boxes输出缓冲区
    output_buf0 = aclCreateDataBuffer(dev_output_box_, output0_size_);
    err = aclmdlAddDatasetBuffer(output_dataset, output_buf0);
    if (check_acl_error(err, "infer-4", "添加pred_boxes输出缓冲区") != 0) throw runtime_error("添加pred_boxes失败");

    // 2.2 添加pred_scores输出缓冲区
    output_buf1 = aclCreateDataBuffer(dev_output_score_, output1_size_);
    err = aclmdlAddDatasetBuffer(output_dataset, output_buf1);
    if (check_acl_error(err, "infer-5", "添加pred_scores输出缓冲区") != 0) throw runtime_error("添加pred_scores失败");

    // 3. 拷贝HOST数据到设备端
    err = aclrtMemcpy(dev_img_t_, input0_size_, host_img_t_, input0_size_, ACL_MEMCPY_HOST_TO_DEVICE);
    if (check_acl_error(err, "infer-6", "拷贝img_t到设备端") != 0) throw runtime_error("拷贝img_t失败");
        
    err = aclrtMemcpy(dev_img_ot_, input1_size_, host_img_ot_, input1_size_, ACL_MEMCPY_HOST_TO_DEVICE);
    if (check_acl_error(err, "infer-7", "拷贝img_ot到设备端") != 0) throw runtime_error("拷贝img_ot失败");
        
    err = aclrtMemcpy(dev_img_search_, input2_size_, host_img_search_, input2_size_, ACL_MEMCPY_HOST_TO_DEVICE);
    if (check_acl_error(err, "infer-8", "拷贝img_search到设备端") != 0) throw runtime_error("拷贝img_search失败");

    // 4. 执行推理（计时放在try块内，避免跨初始化）
    auto start = chrono::steady_clock::now();
    err = aclmdlExecute(model_id_, input_dataset, output_dataset);
    if (check_acl_error(err, "infer-9", "执行模型推理") != 0) throw runtime_error("推理执行失败");

    // 5. 同步流（等待推理完成）
    err = aclrtSynchronizeStream(stream_);
    if (check_acl_error(err, "infer-10", "同步推理流") != 0) throw runtime_error("流同步失败");
        
    auto end = chrono::steady_clock::now();
    double infer_time = chrono::duration<double, milli>(end - start).count();
    cout << "[Mixformer] 推理完成: 耗时=" << infer_time << "ms" << endl;

    // 6. 拷贝输出数据到HOST端
    err = aclrtMemcpy(host_output_box_, output0_size_, dev_output_box_, output0_size_, ACL_MEMCPY_DEVICE_TO_HOST);
    if (check_acl_error(err, "infer-11", "拷贝pred_boxes到HOST端") != 0) throw runtime_error("拷贝pred_boxes失败");
        
    err = aclrtMemcpy(host_output_score_, output1_size_, dev_output_score_, output1_size_, ACL_MEMCPY_DEVICE_TO_HOST);
    if (check_acl_error(err, "infer-12", "拷贝pred_scores到HOST端") != 0) throw runtime_error("拷贝pred_scores失败");

    // 正常流程释放资源
    aclDestroyDataBuffer(input_buf0);
    aclDestroyDataBuffer(input_buf1);
    aclDestroyDataBuffer(input_buf2);
    aclmdlDestroyDataset(input_dataset);
    aclDestroyDataBuffer(output_buf0);
    aclDestroyDataBuffer(output_buf1);
    aclmdlDestroyDataset(output_dataset);

    return 0;
}

// -------------------------- 坐标映射（搜索区域→原图坐标） --------------------------
void Mixformer::map_box_back(DrBBox& pred_box, float resize_factor) {
    // state_是上一帧的目标框（原图坐标）
    float cx_prev = state_.x0 + 0.5 * (state_.x1 - state_.x0);  // 上一帧目标中心x（原图）
    float cy_prev = state_.y0 + 0.5 * (state_.y1 - state_.y0);  // 上一帧目标中心y（原图）
    
    // 搜索区域在原图中的半边长（resize_factor是搜索区域相对于原图的缩放因子）
    float half_search_side = 0.5 * cfg_.search_size / resize_factor;

    // 计算当前帧目标在搜索区域中的中心和宽高（像素坐标）
    float w_search = pred_box.x1 - pred_box.x0;  // 搜索区域中的目标宽
    float h_search = pred_box.y1 - pred_box.y0;  // 搜索区域中的目标高
    float cx_search = pred_box.x0 + 0.5 * w_search;  // 搜索区域中的目标中心x
    float cy_search = pred_box.y0 + 0.5 * h_search;  // 搜索区域中的目标中心y

    // 映射到原图坐标：搜索区域中心对齐上一帧目标中心，再计算当前目标位置
    float cx_real = cx_prev - half_search_side + cx_search / resize_factor;
    float cy_real = cy_prev - half_search_side + cy_search / resize_factor;
    float w_real = w_search / resize_factor;
    float h_real = h_search / resize_factor;

    // 转换为左上角+右下角坐标
    pred_box.x0 = cx_real - 0.5 * w_real;
    pred_box.y0 = cy_real - 0.5 * h_real;
    pred_box.x1 = cx_real + 0.5 * w_real;
    pred_box.y1 = cy_real + 0.5 * h_real;
}

// -------------------------- 边界裁剪 --------------------------
void Mixformer::clip_box(DrBBox& box, int height, int width, int margin) {
    box.x0 = std::min(std::max(0.f, box.x0), (float)(width - margin));
    box.y0 = std::min(std::max(0.f, box.y0), (float)(height - margin));
    box.x1 = std::min(std::max((float)margin, box.x1), (float)width);
    box.y1 = std::min(std::max((float)margin, box.y1), (float)height);
}

// -------------------------- 后处理（解析输出+坐标修正） --------------------------
// int Mixformer::postprocess(float resize_factor, int img_width, int img_height) {
int Mixformer::postprocess(const cv::Mat& img, float resize_factor, int img_width, int img_height) {
    // 解析输出：pred_boxes = [cx, cy, w, h]（归一化到[0,1]）, pred_scores = [conf]
    float cx_norm = host_output_box_[0];
    float cy_norm = host_output_box_[1];
    float w_norm = host_output_box_[2];
    float h_norm = host_output_box_[3];
    float conf = host_output_score_[0];

    // 关键修复：归一化坐标 → 搜索区域像素坐标（搜索区域尺寸=cfg_.search_size=224）
    float cx_pixel = cx_norm * cfg_.search_size;  // 还原为搜索区域的像素cx
    float cy_pixel = cy_norm * cfg_.search_size;  // 还原为搜索区域的像素cy
    float w_pixel = w_norm * cfg_.search_size;    // 还原为搜索区域的像素宽
    float h_pixel = h_norm * cfg_.search_size;    // 还原为搜索区域的像素高

    cout << "[Mixformer] 输出解析: " << endl;
    cout << "  归一化坐标: cx=" << cx_norm << ", cy=" << cy_norm << ", w=" << w_norm << ", h=" << h_norm << endl;
    cout << "  搜索区域像素坐标: cx=" << cx_pixel << ", cy=" << cy_pixel << ", w=" << w_pixel << ", h=" << h_pixel << endl;
    cout << "  置信度: conf=" << conf << endl;

    // 转换为左上角+右下角坐标（基于搜索区域像素坐标）
    DrBBox pred_box;
    pred_box.x0 = cx_pixel - 0.5 * w_pixel;
    pred_box.y0 = cy_pixel - 0.5 * h_pixel;
    pred_box.x1 = cx_pixel + 0.5 * w_pixel;
    pred_box.y1 = cy_pixel + 0.5 * h_pixel;

    // 映射到原图坐标（resize_factor是搜索区域相对于原图的缩放因子）
    map_box_back(pred_box, resize_factor);

    clip_box(pred_box, img_height, img_width, 10);

    // 更新跟踪状态
    object_box_.box = pred_box;
    object_box_.score = conf;
    object_box_.class_id = 0;
    state_ = pred_box;

    // 模板更新逻辑
    max_pred_score_ = max_pred_score_ * max_score_decay_;
    if (conf > 0.5 && conf > max_pred_score_) {
        float tf_resize_factor = 1.f;
        // 临时保留空Mat（后续可优化为传入当前帧图像）
        // sample_target(cv::Mat(), max_oz_patch_, state_, cfg_.template_factor, cfg_.template_size, tf_resize_factor);
        sample_target(img, max_oz_patch_, state_, cfg_.template_factor, cfg_.template_size, tf_resize_factor);
        max_pred_score_ = conf;
    }

    if (frame_id_ % cfg_.update_interval == 0) {
        oz_patch_ = max_oz_patch_.clone();
        max_pred_score_ = -1.f;
    }

    cout << "[Mixformer] 后处理完成: 原图目标框=(" << pred_box.x0 << "," << pred_box.y0 << "," << pred_box.x1 << "," << pred_box.y1 << "), 置信度=" << conf << endl;
    return 0;
}

// -------------------------- 跟踪主函数 --------------------------
const DrOBB& Mixformer::track(const cv::Mat& img) {
    if (!is_inited_) {
        cerr << "[ERROR] 模型未初始化，请先调用init()" << endl;
        return object_box_;
    }
    if (!is_track_inited_) {
        cerr << "[ERROR] 跟踪未初始化，请先调用init_track()" << endl;
        return object_box_;
    }

    frame_id_++;
    float resize_factor = 1.f;
    int img_width = img.cols;
    int img_height = img.rows;

    // 1. 提取搜索区域（img_search）
    cv::Mat x_patch;
    sample_target(img, x_patch, state_, cfg_.search_factor, cfg_.search_size, resize_factor);

    // 2. 预处理（图像→模型输入格式）
    if (preprocess(z_patch_, oz_patch_, x_patch) != 0) {
        cerr << "[ERROR] 预处理失败" << endl;
        return object_box_;
    }

    // 3. 模型推理
    if (infer() != 0) {
        cerr << "[ERROR] 推理失败" << endl;
        return object_box_;
    }

    // 4. 后处理（传入当前帧宽高，优化边界裁剪）
    // if (postprocess(resize_factor, img_width, img_height) != 0) {
    if (postprocess(img, resize_factor, img_width, img_height) != 0) {
        cerr << "[ERROR] 后处理失败" << endl;
        return object_box_;
    }

    return object_box_;
}

// -------------------------- 资源释放 --------------------------
int Mixformer::release() {
    if (!is_inited_) return 0;

    // 释放设备端缓冲区
    if (dev_img_t_ != nullptr) aclrtFree(dev_img_t_);
    if (dev_img_ot_ != nullptr) aclrtFree(dev_img_ot_);
    if (dev_img_search_ != nullptr) aclrtFree(dev_img_search_);
    if (dev_output_box_ != nullptr) aclrtFree(dev_output_box_);
    if (dev_output_score_ != nullptr) aclrtFree(dev_output_score_);

    // 释放模型资源
    if (model_id_ != 0) aclmdlUnload(model_id_);
    if (model_desc_ != nullptr) aclmdlDestroyDesc(model_desc_);
    if (model_mem_ptr_ != nullptr) aclrtFree(model_mem_ptr_);
    if (model_weight_ptr_ != nullptr) aclrtFree(model_weight_ptr_);

    is_inited_ = false;
    is_track_inited_ = false;
    cout << "[INFO] Mixformer资源释放完成!" << endl;
    return 0;
}