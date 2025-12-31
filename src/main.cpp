#include <iostream>
#include <string>
#include <vector>
#include <cstring>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include "acl/acl.h"
#include <chrono> 
#include <sys/stat.h>
#include <sys/types.h>
#include <errno.h>
#include "mixformer.h"

// #define INIT_BBOX Rect(796, 667, 30, 25)        //plan
// #define INIT_BBOX Rect(553, 599, 91, 40)        //Two Uav
// #define INIT_BBOX Rect(1097, 668, 123, 51)           //Tree  
// #define INIT_BBOX Rect(1437, 660, 19, 13)           //24
// #define INIT_BBOX Rect(718, 587, 25, 23)           //25
#define INIT_BBOX Rect(340, 429, 55, 33)           //20


#define DEVICE_ID 0

using namespace std;
using namespace cv;
using namespace std::chrono;  

bool file_exists(const string& path) {
    struct stat buffer;
    return (stat(path.c_str(), &buffer) == 0);
}

int check_acl_error(aclError err, const string& step, const string& msg) {
    if (err != ACL_SUCCESS) {
        cerr << "[ERROR] 步骤[" << step << "] " << msg << " 失败! 错误码:" << err << endl;
        return -1;
    }
    cout << "[INFO] 步骤[" << step << "] " << msg << " 成功!" << endl;
    return 0;
}

int main() {
    string model_path = "../model/mixformer_v2.om";
    // string video_path = "../plane.mp4";
    // string video_path = "../Uav_doublication.mp4";
    // string video_path = "../tree.mp4";
    // string video_path = "../24.mp4";
    // string video_path = "../25.mp4";
    string video_path = "../20.mp4";
    string out_dir = "../out";
    string out_video_path = out_dir + "/track_result.avi";

    cout << "[INFO] 初始边界框配置：" << endl;
    cout << "  左上角(x0,y0): (" << INIT_BBOX.x << ", " << INIT_BBOX.y << ")" << endl;
    cout << "  右下角(x1,y1): (" << INIT_BBOX.x + INIT_BBOX.width << ", " << INIT_BBOX.y + INIT_BBOX.height << ")" << endl;
    cout << "  宽度x高度: " << INIT_BBOX.width << "x" << INIT_BBOX.height << endl;

    if (!file_exists(model_path)) {
        cerr << "[ERROR] 模型文件不存在:" << model_path << endl;
        return -1;
    }
    if (!file_exists(video_path)) {
        cerr << "[ERROR] 视频文件不存在:" << video_path << endl;
        return -1;
    }

    if (mkdir(out_dir.c_str(), 0755) != 0 && errno != EEXIST) {
        cerr << "[ERROR] 创建输出目录失败:" << out_dir << ", 错误码:" << errno << endl;
        return -1;
    }

    // 打开视频
    VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        cerr << "[ERROR] 无法打开视频文件:" << video_path << endl;
        return -1;
    }

    double video_fps = cap.get(CAP_PROP_FPS);
    int frame_width = static_cast<int>(cap.get(CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(cap.get(CAP_PROP_FRAME_HEIGHT));
    int total_frames = static_cast<int>(cap.get(CAP_PROP_FRAME_COUNT));

    cout << "[INFO] 视频参数：" << endl;
    cout << "  帧率: " << video_fps << " FPS" << endl;
    cout << "  帧尺寸: " << frame_width << "x" << frame_height << endl;
    cout << "  预估总帧数: " << total_frames << " 帧" << endl;

    if (INIT_BBOX.x + INIT_BBOX.width > frame_width || INIT_BBOX.y + INIT_BBOX.height > frame_height) {
        cerr << "[WARNING] 初始边界框超出视频帧范围！可能导致跟踪失败" << endl;
        cerr << "  视频帧范围: 0<=x<" << frame_width << ", 0<=y<" << frame_height << endl;
    }

    // 初始化视频写入器
    // int fourcc = VideoWriter::fourcc('M', 'J', 'P', 'G');
    // int fourcc = VideoWriter::fourcc('H', '2', '6', '4');  // 更高效的压缩编码
    int fourcc = VideoWriter::fourcc('X', 'V', 'I', 'D');

    vector<int> encode_params;
    encode_params.push_back(CV_IMWRITE_JPEG_QUALITY);
    encode_params.push_back(75);  // 质量75（推荐值，可根据需求调整）

    VideoWriter out_video;
    out_video.open(out_video_path, fourcc, video_fps, Size(frame_width, frame_height), true);

    if (!out_video.isOpened()) {
        cerr << "[ERROR] 无法创建输出视频:" << out_video_path << endl;
        cap.release();
        return -1;
    }
    cout << "[INFO] 输出视频配置：" << endl;
    cout << "  保存路径: " << out_video_path << endl;
    cout << "  编码格式: MJPG" << endl;
    cout << "  输出帧率: " << video_fps << " FPS" << endl;
    cout << "  输出尺寸: " << frame_width << "x" << frame_height << endl;

    // ACL初始化
    aclError err;
    err = aclInit(nullptr);
    if (check_acl_error(err, "ACL-1", "ACL全局初始化") != 0) {
        cap.release();
        out_video.release();
        return -1;
    }

    err = aclrtSetDevice(DEVICE_ID);
    if (check_acl_error(err, "ACL-2", "打开NPU设备") != 0) {
        aclFinalize();
        cap.release();
        out_video.release();
        return -1;
    }

    aclrtContext context = nullptr;
    err = aclrtCreateContext(&context, DEVICE_ID);
    if (check_acl_error(err, "ACL-3", "创建ACL上下文") != 0) {
        aclrtResetDevice(DEVICE_ID);
        aclFinalize();
        cap.release();
        out_video.release();
        return -1;
    }

    aclrtStream stream = nullptr;
    err = aclrtCreateStream(&stream);
    if (check_acl_error(err, "ACL-4", "创建ACL流") != 0) {
        aclrtDestroyContext(context);
        aclrtResetDevice(DEVICE_ID);
        aclFinalize();
        cap.release();
        out_video.release();
        return -1;
    }

    // Mixformer初始化
    Mixformer mixformer(model_path, context, stream);
    if (mixformer.init() != 0) {
        cerr << "[ERROR] Mixformer初始化失败!" << endl;
        mixformer.release();
        aclrtDestroyStream(stream);
        aclrtDestroyContext(context);
        aclrtResetDevice(DEVICE_ID);
        aclFinalize();
        cap.release();
        out_video.release();
        return -1;
    }

    // 跟踪初始化（第一帧）
    DrOBB init_obb;
    init_obb.box.x0 = INIT_BBOX.x;
    init_obb.box.y0 = INIT_BBOX.y;
    init_obb.box.x1 = INIT_BBOX.x + INIT_BBOX.width;
    init_obb.box.y1 = INIT_BBOX.y + INIT_BBOX.height;
    init_obb.score = 0.9f;
    init_obb.class_id = 0;

    Mat first_img;
    if (!cap.read(first_img)) {
        cerr << "[ERROR] 无法读取视频第一帧!" << endl;
        mixformer.release();
        aclrtDestroyStream(stream);
        aclrtDestroyContext(context);
        aclrtResetDevice(DEVICE_ID);
        aclFinalize();
        cap.release();
        out_video.release();
        return -1;
    }
    mixformer.init_track(first_img, init_obb);

    // 帧处理统计（修改：仅统计「取帧→推理」耗时）
    auto total_start = steady_clock::now();
    int frame_count = 0;          // 已处理帧数
    int valid_frame_count = 0;    // 有效处理帧数
    double total_infer_time_ms = 0.0;  // 仅统计：取帧→推理完成的总耗时

    cout << "\n[INFO] 开始处理视频帧..." << endl;
    while (true) {
        Mat img;
        if (!cap.read(img)) {
            cout << "[INFO] 视频帧读取完成（或遇到错误）" << endl;
            break;
        }
        frame_count++;
        cout << "\n====================================\n";
        cout << "[INFO] 处理第" << frame_count << "帧" << endl;

        // -------------------------- 计时开始：取到帧后，推理前 --------------------------
        auto infer_start = steady_clock::now();

        // 执行跟踪（核心推理逻辑）
        const DrOBB& result = mixformer.track(img);

        // -------------------------- 计时结束：推理完成后 --------------------------
        auto infer_end = steady_clock::now();
        double infer_time_ms = duration_cast<milliseconds>(infer_end - infer_start).count();
        total_infer_time_ms += infer_time_ms;
        valid_frame_count++;

        // 绘制框和写入视频（不纳入统计）
        Rect track_rect(
            static_cast<int>(result.box.x0), 
            static_cast<int>(result.box.y0), 
            static_cast<int>(result.box.x1 - result.box.x0), 
            static_cast<int>(result.box.y1 - result.box.y0)
        );
        rectangle(img, track_rect, Scalar(0, 255, 0), 2);
        string label = "Conf: " + to_string(result.score).substr(0, 4);
        putText(img, label, Point(track_rect.x, track_rect.y - 5), 
                FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 1);
        out_video.write(img);
        cout << "[INFO] 第" << frame_count << "帧已写入输出视频" << endl;

        // 基于纯推理耗时计算帧率
        double infer_fps = 1000.0 / infer_time_ms;
        cout << "[INFO] 第" << frame_count << "帧推理耗时: " << infer_time_ms << "ms | 推理帧率: " << infer_fps << "FPS" << endl;
    }

    // 统计报告（基于纯推理耗时）
    auto total_end = steady_clock::now();
    double total_time_sec = duration_cast<milliseconds>(total_end - total_start).count() / 1000.0;
    double total_infer_time_sec = total_infer_time_ms / 1000.0;
    double avg_infer_fps = valid_frame_count / total_infer_time_sec;  // 平均推理帧率
    double overall_fps = valid_frame_count / total_time_sec;         // 整体帧率（含绘制/保存）
    double avg_infer_time_ms = total_infer_time_ms / valid_frame_count;  // 平均推理耗时

    cout << "\n====================================\n";
    cout << "[INFO] 帧率统计报告（仅统计推理耗时）:" << endl;
    cout << "  总处理帧数: " << frame_count << "帧" << endl;
    cout << "  有效推理帧数: " << valid_frame_count << "帧" << endl;
    cout << "  视频原始帧率: " << video_fps << "FPS" << endl;
    cout << "  总耗时（含绘制/保存）: " << total_time_sec << "秒" << endl;
    cout << "  纯推理总耗时: " << total_infer_time_sec << "秒" << endl;
    cout << "  平均推理帧率: " << avg_infer_fps << "FPS" << endl;  // 核心统计项
    cout << "  平均推理耗时: " << avg_infer_time_ms << "ms" << endl;  // 核心统计项
    cout << "  整体帧率（含绘制/保存）: " << overall_fps << "FPS" << endl;  // 参考项
    cout << "====================================\n";

    // 资源释放
    mixformer.release();
    aclrtDestroyStream(stream);
    aclrtDestroyContext(context);
    aclrtResetDevice(DEVICE_ID);
    aclFinalize();
    cap.release();
    out_video.release();

    cout << "\n[INFO] 所有处理完成！" << endl;
    cout << "  输入视频: " << video_path << endl;
    cout << "  输出视频: " << out_video_path << endl;
    return 0;
}