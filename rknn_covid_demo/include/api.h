//结果结构体，都是左上角坐标系
struct DetectRes {
    int face_x; //脸部坐标
    int face_y; 
    int face_width; //脸部框的宽高
    int face_height; 
    float face_detect_conf; //脸部检测框的置信度（分数score）
    
    float face_pts[68][2]; //该二维数组保存了 脸部68个点的x,y坐标
    
    int mouth_x; //嘴部坐标
    int mouth_y;
    int mouth_width; //嘴部框的宽高
    int mouth_height;
    
    float open_ratio; //嘴巴张开程度
    
    int l_x; //左侧腺体框检测区域坐标
    int l_y;
    int l_width;
    int l_height;
    
    int r_x; //右侧腺体框检测区域坐标
    int r_y; 
    int r_width;
    int r_height;
    
    float l_gray_val;//左侧腺体框检测结果值
    float r_gray_val;//右侧腺体框检测结果值
    
    DetectRes(){ //赋予变量初始值来初始化所有变量
        face_x = 0;
        face_y = 0;
        face_width = 0;
        face_height = 0;
        face_detect_conf = 0.0;
        mouth_x = 0;
        mouth_y = 0;
        mouth_width = 0;
        mouth_height = 0;
        open_ratio = 0.0;
        l_x = 0;
        l_y = 0;
        l_width = 0;
        l_height = 0;
        r_x = 0;
        r_y = 0;
        r_width = 0;
        r_height = 0;
        l_gray_val = 0.0;
        r_gray_val = 0.0;
    }
};

//参数结构体
struct DetectParam {
	int detect_area_x; //检测区域左上角x
	int detect_area_y; //检测区域左上角y
	int detect_area_width; //检测区域宽度
	int detect_area_height; //检测区域高度
	float iou_th; //嘴部是否到位检测
	float open_ratio; //嘴部张开的比例
	int frame_th;  //连续帧阈值
	float inner_ratio; //收缩比例
	int stick_stay_th; //棉签检测阈值
	float conf_th; //灰度阈值
};

// 初始化环境，加载2个AI模型
// return type: 0 => success, -1 => 环境初始化失败
extern "C" int InitEnv(const char* model_path1, const char* model_path2); 

// 循环调用该函数，对每帧数据执行检测
// 结果返回类型
// -1 图像数据为空
// 0 检测完成
// 1 未检测到面部
// 2 未将嘴部放入待检测区域
// 3 嘴部张开幅度不够大
// 4 棉签等待时间不够久
extern "C" int DoDetection(const unsigned char* img, int width, int height, DetectParam param, DetectRes& detect_res);

// 释放模型资源
// return type: 0 => success, -1 => 释放失败
extern "C" int ReleaseEnv();
