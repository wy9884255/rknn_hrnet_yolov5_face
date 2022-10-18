#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <deque>
#include <numeric>
#include <sys/time.h>
#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "postprocess.h"
#include "rknn_api.h"
#include "api.h"

void dump_tensor_attr(rknn_tensor_attr* attr)
{
  printf("  index=%d, name=%s, n_dims=%d, dims=[%d, %d, %d, %d], n_elems=%d, size=%d, fmt=%s, type=%s, qnt_type=%s, "
         "zp=%d, scale=%f\n",
         attr->index, attr->name, attr->n_dims, attr->dims[0], attr->dims[1], attr->dims[2], attr->dims[3],
         attr->n_elems, attr->size, get_format_string(attr->fmt), get_type_string(attr->type),
         get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}

double __get_us(struct timeval t) { return (t.tv_sec * 1000000 + t.tv_usec); }

unsigned char* load_data(FILE* fp, size_t ofst, size_t sz)
{
  unsigned char* data;
  int            ret;

  data = NULL;

  if (NULL == fp) {
    return NULL;
  }

  ret = fseek(fp, ofst, SEEK_SET);
  if (ret != 0) {
    printf("blob seek failure.\n");
    return NULL;
  }

  data = (unsigned char*)malloc(sz);
  if (data == NULL) {
    printf("buffer malloc failure.\n");
    return NULL;
  }
  ret = fread(data, 1, sz, fp);
  return data;
}

unsigned char* load_model(const char* filename, int* model_size)
{
  FILE*          fp;
  unsigned char* data;

  fp = fopen(filename, "rb");
  if (NULL == fp) {
    printf("Open file %s failed.\n", filename);
    return NULL;
  }

  fseek(fp, 0, SEEK_END);
  int size = ftell(fp);

  data = load_data(fp, 0, size);

  fclose(fp);

  *model_size = size;
  return data;
}

//for hrnet preprocses from here
int bbox_xywh2cs(cv::Rect& bbox, cv::Point2f& center, cv::Point2f& scale, float aspect_ratio, float padding, float pixel_std){
	float x = bbox.x;
	float y = bbox.y;
	float w = bbox.width;
	float h = bbox.height;
	center = cv::Point2f(w / 2.0, h / 2.0);
	if (w > aspect_ratio * h){
		h = w * 1.0 / aspect_ratio;
	}else if(w < aspect_ratio * h){
		w = h * aspect_ratio;
	}
	scale = cv::Point2f(w / pixel_std, h / pixel_std);
	scale.x *= padding;
	scale.y *= padding;
	return 0;
}

int get_affine_transform(cv::Point2f& center, cv::Point2f& scale, cv::Size output_size, cv::Mat& trans){
	cv::Point2f scale_temp(scale.x * 200.0, scale.y * 200.0);
	cv::Point2f shift(0.0, 0.0);
	float src_w = scale_temp.x;
	float dst_w = output_size.width;
	float dst_h = output_size.height;
	
	cv::Point2f src_dir(0.0, src_w * -0.5);
	cv::Point2f dst_dir(0.0, dst_w * -0.5);
	
	cv::Point2f src[3];
	src[0] = cv::Point2f(center.x + scale_temp.x * shift.x, center.y + scale_temp.y * shift.y);
	src[1] = cv::Point2f(center.x + src_dir.x + scale_temp.x * shift.x, center.y + src_dir.y + scale_temp.y * shift.y);
	cv::Point2f d1(src[0].x - src[1].x, src[0].y - src[1].y);
	src[2] = cv::Point2f(src[1].x - d1.y, src[1].y + d1.x);
	
	cv::Point2f dst[3];
	dst[0] = cv::Point2f(dst_w * 0.5, dst_h * 0.5);
	dst[1] = cv::Point2f(dst_w * 0.5 + dst_dir.x, dst_h * 0.5 + dst_dir.y);
	cv::Point2f d2(dst[0].x - dst[1].x, dst[0].y - dst[1].y);
	dst[2] = cv::Point2f(dst[1].x - d2.y, dst[1].y + d2.x);
	cv::Mat t = cv::getAffineTransform(src, dst);
	trans = t;
	return 0;
}

//for hrnet post process from here
int get_max_preds(float* output_ptr, std::vector<cv::Point2f>& preds, std::vector<float>& maxvals){
	for(int i = 0; i < 68; ++i){
		float max_val = -1000000.0;
		int max_col = -1;
		int max_row = -1;
		for(int j = 0; j < 64; ++j){
			for(int k = 0; k < 64; ++k){
				float val = output_ptr[i*64*64 + j*64 + k];
				if (val > max_val){
					max_val = val;
					max_col = j;
					max_row = k;
				}
			}
		}
		//per cls
		maxvals.push_back(max_val);
		if (max_val < 0){
			preds.push_back(cv::Point2f(-1.0, -1.0));
		}else{
			preds.push_back(cv::Point2f(max_row, max_col));
		}	
		//reinit
		max_val = INT_MIN;
		max_col = -1;
		max_row = -1;		
	}
}

int transform_preds(std::vector<cv::Point2f>& preds, cv::Point2f center, cv::Point2f scale){
	scale.x *= 200.0;
	scale.y *= 200.0;
	float scale_x = scale.x / 64.0;
	float scale_y = scale.y / 64.0;
	for (int i = 0; i < 68; ++i){
		preds[i].x = preds[i].x * scale_x + center.x - scale.x * 0.5;
		preds[i].y = preds[i].y * scale_y + center.y - scale.y * 0.5;
	}
}

int keypoints_from_heatmaps(float* output_ptr, cv::Point2f center, cv::Point2f scale, 
std::vector<cv::Point2f>& preds, std::vector<float>& maxvals){
	get_max_preds(output_ptr, preds, maxvals);
	for (int i = 0; i < 68; ++i){
		int px = int(preds[i].x);
		int py = int(preds[i].y);
		if ((px > 1 && px < 64 - 1) && (py > 1 && py < 64-1)){
			float diff_x = output_ptr[i*64*64 + py * 64 + px + 1] - output_ptr[i*64*64 + py * 64 + px - 1];
			float diff_y = output_ptr[i*64*64 + (py + 1) * 64 + px] - output_ptr[i*64*64 + (py - 1) * 64 + px];
			if (diff_x > 0){
				preds[i].x += 1 * 0.25;
			}else if(diff_x < 0) {
				preds[i].x += -1 * 0.25;
			}
			if (diff_y > 0){
				preds[i].y += 1 * 0.25;
			}else if(diff_y < 0) {
				preds[i].y += -1 * 0.25;
			}
		}
	}
	transform_preds(preds, center, scale);
}

float euclideanDist(cv::Point2f& a, cv::Point2f& b)
{
    cv::Point2f diff = a - b;
    return cv::sqrt(diff.x*diff.x + diff.y*diff.y);
}

rknn_context ctx1;
rknn_context ctx2;
unsigned char* model_data1 = nullptr;
unsigned char* model_data2 = nullptr;
rknn_input_output_num io_num1;
rknn_input_output_num io_num2;
std::deque<float> dq_x1;
std::deque<float> dq_y1;
std::deque<float> dq_x2;
std::deque<float> dq_y2;
std::deque<float> dq1;//处理左侧连续帧检测结果
std::deque<float> dq2;//处理右侧连续帧检测结果

void reset_containers(){
	dq_x1.clear();
	dq_y2.clear();
	dq_x2.clear();
	dq_y2.clear();
	dq1.clear();
	dq2.clear();
}


// 初始化环境，加载2个AI模型
int InitEnv(const char* model_path1, const char* model_path2){
  //load yolo model
  printf("Loading yolo mode...\n");
  int            model_data_size1 = 0;
  unsigned char* model_data1      = load_model(model_path1, &model_data_size1);
  int ret                         = rknn_init(&ctx1, model_data1, model_data_size1, 0, NULL);
  if (ret < 0) {
    printf("yolo rknn_init error ret=%d\n", ret);
    return -1;
  }
  printf("yolo rknn_init success!\n", ret);
  
  //load hrnet model
  printf("Loading hrnet mode...\n");
  int            model_data_size2 = 0;
  unsigned char* model_data2      = load_model(model_path2, &model_data_size2);
  ret                            = rknn_init(&ctx2, model_data2, model_data_size2, 0, NULL);
  if (ret < 0) {
    printf("hrnet rknn_init error ret=%d\n", ret);
    return -1;
  }
  printf("hrnet rknn_init success!\n", ret);
  
  rknn_sdk_version version;
  ret = rknn_query(ctx1, RKNN_QUERY_SDK_VERSION, &version, sizeof(rknn_sdk_version));
  if (ret < 0) {
    printf("rknn_init error ret=%d\n", ret);
    return -1;
  }
  printf(" enviroment init success, sdk version: %s driver version: %s\n", version.api_version, version.drv_version);
  return 0;
}

//yolov5检测人脸坐标
int YoloDetection(const cv::Mat& src, DetectRes& detect_res){

  cv::Mat img;
  cv::cvtColor(src, img, cv::COLOR_BGR2RGB);
  int img_width1  = img.cols;
  int img_height1 = img.rows;
  int ret = rknn_query(ctx1, RKNN_QUERY_IN_OUT_NUM, &io_num1, sizeof(io_num1));
  struct timeval start_time, stop_time;
  if (ret < 0) {
    printf("rknn_init error ret=%d\n", ret);
    return -1;
  }
  //printf("yolo model input num: %d, output num: %d\n", io_num1.n_input, io_num1.n_output);

  rknn_tensor_attr input_attrs1[io_num1.n_input];
  memset(input_attrs1, 0, sizeof(input_attrs1));
  for (int i = 0; i < io_num1.n_input; i++) {
    input_attrs1[i].index = i;
    ret                  = rknn_query(ctx1, RKNN_QUERY_INPUT_ATTR, &(input_attrs1[i]), sizeof(rknn_tensor_attr));
    if (ret < 0) {
      printf("rknn_init error ret=%d\n", ret);
      return -1;
    }
    //dump_tensor_attr(&(input_attrs1[i]));
  }

  rknn_tensor_attr output_attrs1[io_num1.n_output];
  memset(output_attrs1, 0, sizeof(output_attrs1));
  for (int i = 0; i < io_num1.n_output; i++) {
    output_attrs1[i].index = i;
    ret                   = rknn_query(ctx1, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs1[i]), sizeof(rknn_tensor_attr));
    //dump_tensor_attr(&(output_attrs1[i]));
  }
  
  int channel1 = 3;
  int width1   = 0;
  int height1  = 0;
  if (input_attrs1[0].fmt == RKNN_TENSOR_NCHW) {
    //printf("yolo model is NCHW input fmt\n");
    channel1 = input_attrs1[0].dims[1];
    width1   = input_attrs1[0].dims[2];
    height1  = input_attrs1[0].dims[3];
  } else {
    //printf("yolo model is NHWC input fmt\n");
    width1   = input_attrs1[0].dims[1];
    height1  = input_attrs1[0].dims[2];
    channel1 = input_attrs1[0].dims[3];
  }
  //printf("yolo model input height=%d, width=%d, channel=%d\n", height1, width1, channel1);

  rknn_input inputs1[1];
  memset(inputs1, 0, sizeof(inputs1));
  inputs1[0].index        = 0;
  inputs1[0].type         = RKNN_TENSOR_UINT8;
  inputs1[0].size         = width1 * height1 * channel1;
  inputs1[0].fmt          = RKNN_TENSOR_NHWC;
  inputs1[0].pass_through = 0;
  
  // yolov5 input 640*640
  if (img_width1 != width1 || img_height1 != height1) {
    cv::Mat resized_img;
	  cv::resize(img, resized_img, cv::Size(width1, height1));
    int all = resized_img.channels() * resized_img.cols * resized_img.rows;
    uint8_t *d2 = new uint8_t[all];

    for (int i = 0; i < resized_img.rows; ++i) {
        for (int j = 0; j < resized_img.cols; ++j) {
            cv::Vec3b vc = resized_img.at<cv::Vec3b>(i, j);
            d2[i*resized_img.cols*3 + j * 3 + 0] = (uint8_t)vc.val[0];
            d2[i*resized_img.cols*3 + j * 3 + 1] = (uint8_t)vc.val[1];
            d2[i*resized_img.cols*3 + j * 3 + 2] = (uint8_t)vc.val[2];
        }
    }
    inputs1[0].buf = d2;
  } else {
    inputs1[0].buf = (void*)img.data;
  }
  
  gettimeofday(&start_time, NULL);
  rknn_inputs_set(ctx1, io_num1.n_input, inputs1);

  rknn_output outputs1[io_num1.n_output];
  memset(outputs1, 0, sizeof(outputs1));
  for (int i = 0; i < io_num1.n_output; i++) {
    outputs1[i].want_float = 0;
  }

  ret = rknn_run(ctx1, NULL);
  ret = rknn_outputs_get(ctx1, io_num1.n_output, outputs1, NULL);
  
  // yolo post process
  float scale_w = (float)width1 / img_width1;
  float scale_h = (float)height1 / img_height1;

  detect_result_group_t detect_result_group;
  std::vector<float>    out_scales1;
  std::vector<int32_t>  out_zps1;
  for (int i = 0; i < io_num1.n_output; ++i) {
    out_scales1.push_back(output_attrs1[i].scale);
    out_zps1.push_back(output_attrs1[i].zp);
  }
	  
  post_process((int8_t*)outputs1[0].buf, (int8_t*)outputs1[1].buf, (int8_t*)outputs1[2].buf, height1, width1,
               BOX_THRESH, NMS_THRESH, scale_w, scale_h, out_zps1, out_scales1, &detect_result_group);
  if (detect_result_group.count == 0){return 1;}
  
  //top1 face only
  char text[256];
  detect_result_t* det_result = &(detect_result_group.results[0]);
  sprintf(text, "%s %.1f%%", det_result->name, det_result->prop * 100);
  printf("%s @ (%d %d %d %d) %f\n", det_result->name, det_result->box.left, det_result->box.top,
           det_result->box.right, det_result->box.bottom, det_result->prop);
  detect_res.face_x = det_result->box.left;
  detect_res.face_y = det_result->box.top;
  detect_res.face_width = det_result->box.right - det_result->box.left;
  detect_res.face_height = det_result->box.bottom - det_result->box.top;
  detect_res.face_detect_conf = det_result->prop;
  
  //release memory
  gettimeofday(&stop_time, NULL);
  ret = rknn_outputs_release(ctx1, io_num1.n_output, outputs1);
  printf("yolo once run use %f ms\n", (__get_us(stop_time) - __get_us(start_time)) / 1000);
  return ret;
}

//hrnet检测人脸关键点
int HRNetDetection(const cv::Mat& src, DetectRes& detect_res){
  struct timeval start_time, stop_time;
  cv::Rect roi(detect_res.face_x, detect_res.face_y, detect_res.face_width, detect_res.face_height);
  roi &= cv::Rect(0, 0, src.cols, src.rows);//求交集
  if (roi.width == 0) return 1; 
  cv::Mat img = src.clone();
  img = img(roi).clone();
  cv::Point2f center, scale;
  bbox_xywh2cs(roi, center, scale, 1.0, 1.25, 200);
  cv::Mat trans;
  get_affine_transform(center, scale, cv::Size(256, 256), trans);
  cv::Mat trans_img;
  cv::warpAffine(img, trans_img, trans, cv::Size(256, 256), cv::INTER_LINEAR);
  //printf("hrnet img preprocess done for keypoint detection\n");

  int ret = rknn_query(ctx2, RKNN_QUERY_IN_OUT_NUM, &io_num2, sizeof(io_num2));
  if (ret < 0) {
    printf("hrnet rknn_init error ret=%d\n", ret);
    return -1;
  }
  //printf("hrnet model input num: %d, output num: %d\n", io_num2.n_input, io_num2.n_output);

  rknn_tensor_attr input_attrs2[io_num2.n_input];
  memset(input_attrs2, 0, sizeof(input_attrs2));
  for (int i = 0; i < io_num2.n_input; i++) {
    input_attrs2[i].index = i;
    ret                  = rknn_query(ctx2, RKNN_QUERY_INPUT_ATTR, &(input_attrs2[i]), sizeof(rknn_tensor_attr));
    if (ret < 0) {
      printf("rknn_init error ret=%d\n", ret);
      return -1;
    }
    //dump_tensor_attr(&(input_attrs2[i]));
  }

  rknn_tensor_attr output_attrs2[io_num2.n_output];
  memset(output_attrs2, 0, sizeof(output_attrs2));
  for (int i = 0; i < io_num2.n_output; i++) {
    output_attrs2[i].index = i;
    ret                   = rknn_query(ctx2, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs2[i]), sizeof(rknn_tensor_attr));
    //dump_tensor_attr(&(output_attrs2[i]));
  }

  int channel2 = 3;
  int width2   = 0;
  int height2  = 0;
  if (input_attrs2[0].fmt == RKNN_TENSOR_NCHW) {
    //printf("hrnet model is NCHW input fmt\n");
    channel2 = input_attrs2[0].dims[1];
    width2   = input_attrs2[0].dims[2];
    height2  = input_attrs2[0].dims[3];
  } else {
    //printf("hrnet model is NHWC input fmt\n");
    width2   = input_attrs2[0].dims[1];
    height2  = input_attrs2[0].dims[2];
    channel2 = input_attrs2[0].dims[3];
  }

  //printf("hrnet model input height=%d, width=%d, channel=%d\n", height2, width2, channel2);

  rknn_input inputs2[1];
  memset(inputs2, 0, sizeof(inputs2));
  inputs2[0].index        = 0;
  inputs2[0].type         = RKNN_TENSOR_UINT8;
  inputs2[0].size         = width2 * height2 * channel2;
  inputs2[0].fmt          = RKNN_TENSOR_NHWC;
  inputs2[0].pass_through = 0;
  inputs2[0].buf = (void*)trans_img.data;
  
  gettimeofday(&start_time, NULL);
  rknn_inputs_set(ctx2, io_num2.n_input, inputs2);

  rknn_output outputs2[io_num2.n_output];
  memset(outputs2, 0, sizeof(outputs2));
  for (int i = 0; i < io_num2.n_output; i++) {
    outputs2[i].want_float = 0;
  }

  ret = rknn_run(ctx2, NULL);
  ret = rknn_outputs_get(ctx2, io_num2.n_output, outputs2, NULL);
  
  //heatmap output, post process
  int8_t* output_ptr = (int8_t*)outputs2[0].buf;
  std::vector<float> out_scales2;
  std::vector<int32_t> out_zps2;
  for (int i = 0; i < io_num2.n_output; ++i) {
    out_scales2.push_back(output_attrs2[i].scale);
    out_zps2.push_back(output_attrs2[i].zp);
  }
  float* output_fl32 = new float[outputs2[0].size];
  for (int i = 0; i < outputs2[0].size; ++i){
	output_fl32[i] = deqnt_affine_to_f32(*(output_ptr + i), out_zps2[0], out_scales2[0]);
  }
  std::vector<cv::Point2f> preds;
  std::vector<float> maxvals;
  keypoints_from_heatmaps(output_fl32, center, scale, preds, maxvals);
  gettimeofday(&stop_time, NULL);
  printf("hrnet once run use %f ms\n", (__get_us(stop_time) - __get_us(start_time)) / 1000);
  for(int i = 0; i < preds.size(); ++i){
	detect_res.face_pts[i][0] = preds[i].x + roi.x;
	detect_res.face_pts[i][1] = preds[i].y + roi.y;
  }
  ret = rknn_outputs_release(ctx2, io_num2.n_output, outputs2);
  return ret;
}

//核酸检测流程判断
int CovidDetection(const cv::Mat& src, const DetectParam& param, DetectRes& detect_res){
	cv::Rect area1(param.detect_area_x, param.detect_area_y, param.detect_area_width, param.detect_area_height);
	cv::Point2f tl = cv::Point2f(detect_res.face_pts[48][0], detect_res.face_pts[50][1]);
	cv::Point2f rd = cv::Point2f(detect_res.face_pts[54][0], detect_res.face_pts[56][1]);
	cv::Rect area2(tl, rd);
	
	detect_res.mouth_x = tl.x;
	detect_res.mouth_y = tl.y;
	detect_res.mouth_width = rd.x - tl.x;
	detect_res.mouth_height = rd.y - tl.y;
	
	cv::Rect iou = area1 | area2;
	float iou_f = iou.area() / area2.area();
	//判断嘴部是否在检测区域内
	if(iou_f < param.iou_th){
		reset_containers();
		return 2;
	}
	
	//检测嘴部张开程度是否合规
	//第一组
	cv::Point2f pt50 = cv::Point2f(detect_res.face_pts[50][0], detect_res.face_pts[50][1]);
	cv::Point2f pt51 = cv::Point2f(detect_res.face_pts[51][0], detect_res.face_pts[51][1]);
	cv::Point2f pt52 = cv::Point2f(detect_res.face_pts[52][0], detect_res.face_pts[52][1]);
	
	cv::Point2f pt58 = cv::Point2f(detect_res.face_pts[58][0], detect_res.face_pts[58][1]);
	cv::Point2f pt57 = cv::Point2f(detect_res.face_pts[57][0], detect_res.face_pts[57][1]);
	cv::Point2f pt56 = cv::Point2f(detect_res.face_pts[56][0], detect_res.face_pts[56][1]);

	//第二组
	cv::Point2f pt61 = cv::Point2f(detect_res.face_pts[61][0], detect_res.face_pts[61][1]);
	cv::Point2f pt62 = cv::Point2f(detect_res.face_pts[62][0], detect_res.face_pts[62][1]);
	cv::Point2f pt63 = cv::Point2f(detect_res.face_pts[63][0], detect_res.face_pts[63][1]);
	
	cv::Point2f pt67 = cv::Point2f(detect_res.face_pts[67][0], detect_res.face_pts[67][1]);
	cv::Point2f pt66 = cv::Point2f(detect_res.face_pts[66][0], detect_res.face_pts[66][1]);
	cv::Point2f pt65 = cv::Point2f(detect_res.face_pts[65][0], detect_res.face_pts[65][1]);
	
	//outside
	float pair1 = euclideanDist(pt50, pt58);
	float pair2 = euclideanDist(pt51, pt57);
	float pair3 = euclideanDist(pt52, pt56);
	//inside
	float pair4 = euclideanDist(pt61, pt67);
	float pair5 = euclideanDist(pt62, pt66);
	float pair6 = euclideanDist(pt63, pt65);
	
	float outside_mean = (pair1 + pair2 + pair3) / 3.0;
	float inside_mean = (pair4 + pair5 + pair6) / 3.0;
	
	float out_ratio = inside_mean / outside_mean;
	detect_res.open_ratio = out_ratio;
	if (out_ratio < param.open_ratio){
		reset_containers();
		return 3;
	}
	
	//检测棉签是否到位
	cv::Point2f pt48 = cv::Point2f(detect_res.face_pts[48][0], detect_res.face_pts[48][1]);
	cv::Point2f pt54 = cv::Point2f(detect_res.face_pts[54][0], detect_res.face_pts[54][1]);
	dq_x1.push_back(pt48.x);
	dq_y1.push_back(pt48.y);
	dq_x2.push_back(pt54.x);
	dq_y2.push_back(pt54.y);
	if (dq_x1.size() > param.frame_th){
		dq_x1.pop_front();
		dq_y1.pop_front();
		dq_x2.pop_front();
		dq_y2.pop_front();
		float mean_x1 = std::accumulate(dq_x1.begin(), dq_x1.end(), 0.0) / (dq_x1.size() * 1.0);
		float mean_y1 = std::accumulate(dq_y1.begin(), dq_y1.end(), 0.0) / (dq_y1.size() * 1.0);
		float mean_x2 = std::accumulate(dq_x2.begin(), dq_x2.end(), 0.0) / (dq_x2.size() * 1.0);
		float mean_y2 = std::accumulate(dq_y2.begin(), dq_y2.end(), 0.0) / (dq_y2.size() * 1.0);
		float mouth_dis_hor = (mean_x2 - mean_x1) / 10.0;
		//display only
		cv::Rect left_roi(cv::Point(mean_x1 + mouth_dis_hor, mean_y1 - mouth_dis_hor), 
		cv::Point(mean_x1 + 2.0 * mouth_dis_hor, mean_y1));
		cv::Rect right_roi(cv::Point(mean_x2 - 2.0 * mouth_dis_hor, mean_y2 - mouth_dis_hor),
		cv::Point(mean_x2 - mouth_dis_hor, mean_y2));
		
		detect_res.l_x = left_roi.x;
		detect_res.l_y = left_roi.y;
		detect_res.l_width = left_roi.width;
		detect_res.l_height = left_roi.height;
		detect_res.r_x = right_roi.x;
		detect_res.r_y = right_roi.y;
		detect_res.r_width = right_roi.width;
		detect_res.r_height = right_roi.height;
		
		//detect usage
		float inner_ratio = param.inner_ratio;
		cv::Rect left_inner_roi(left_roi.x + left_roi.width * inner_ratio, left_roi.y, 
		left_roi.width*inner_ratio*2.0, left_roi.height*inner_ratio*2.0);
		cv::Rect right_inner_roi(right_roi.x + right_roi.width * inner_ratio, right_roi.y, 
		right_roi.width*inner_ratio*2.0, right_roi.height*inner_ratio*2.0);
		left_inner_roi &= cv::Rect(0, 0, src.cols, src.rows);
		if (left_inner_roi.width == 0) return 4; 
		right_inner_roi &= cv::Rect(0, 0, src.cols, src.rows);
		if (right_inner_roi.width == 0) return 4; 
		cv::Mat left_region = src(left_inner_roi).clone();
		cv::Mat right_region = src(right_inner_roi).clone();
		cv::Mat left_region_gary;
		cv::Mat right_region_gary;
		cv::cvtColor(left_region, left_region_gary, cv::COLOR_BGR2GRAY);
		cv::cvtColor(right_region, right_region_gary, cv::COLOR_BGR2GRAY);
		cv::Scalar l_mean = cv::mean(left_region_gary);
		cv::Scalar r_mean = cv::mean(right_region_gary);
		float left_mean = l_mean[0];
		float right_mean = r_mean[0];
		printf("left mean %f, right mean %f\n", left_mean, right_mean);
		detect_res.l_gray_val = left_mean;
		detect_res.r_gray_val = right_mean;
		if (left_mean >= param.conf_th){
			if (dq1.size() < param.stick_stay_th){
			    dq1.push_back(1);
			}
		}else if (right_mean >= param.conf_th){
			if (dq2.size() < param.stick_stay_th){
			    dq2.push_back(1);
			}
		}
		if (dq1.size() == param.stick_stay_th && dq2.size() == param.stick_stay_th){
			reset_containers();
			return 0;
		}else{
			return 4;
		}
	}else{
		return 4;
	}		
	return 0;
}

//循环调用该函数，对每帧数据执行检测
int DoDetection(const unsigned char* img, int width, int height, DetectParam param, DetectRes& detect_res){
	cv::Mat src = cv::Mat(height, width, CV_8UC3, (void*)img);
	if (src.empty()){
		printf("source img is empty\n");
		return -1;
	}
	int ret = YoloDetection(src, detect_res);
	if (ret != 0) return ret;
	ret = HRNetDetection(src, detect_res);
	if (ret != 0) return ret;
	ret = CovidDetection(src, param, detect_res);
	return ret;
}

//释放内存
int ReleaseEnv(){
  int ret = 0;

  ret = rknn_destroy(ctx1);
  ret = rknn_destroy(ctx2);

  if (model_data1) {
    free(model_data1);
  }
  if (model_data2) {
    free(model_data2);
  }
  return 0;
}