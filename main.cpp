#include <opencv4/opencv2/opencv.hpp>
#include <iostream>
#include <string>

const std::string PATH = "/home/minhpkm/FPT/Training/homework/ass_oct/MaskDetect/";
const std::string CASCADE_PATH = PATH + "haarcascade_frontalface_default.xml";
const std::string INPUT_DIR = PATH + "data/input";
const std::string OUTPUT_DIR = PATH + "data/output";

void detectImage(cv::Mat& image, cv::CascadeClassifier& face_cascade){
    cv::Mat gray_image;
    cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);

    std::vector<cv::Rect> faces;
    face_cascade.detectMultiScale(gray_image, faces);

    for(const auto &face : faces){
        cv::rectangle(image, face, cv::Scalar(255,0,0),2);
    }

}


int main(int argc, char** argv){

    if(argc < 2){
        std::cout << "Please provide the image!\n";
        return -1;
    }

    std::string imagePath = INPUT_DIR + argv[1];
    cv::Mat image = cv::imread(imagePath);
    if(image.empty()){
        std::cout << "Cannot open the image!\n";
        return -1;
    }

    cv::CascadeClassifier face_cascade;
    if(!face_cascade.load(CASCADE_PATH)){
        std::cout << "Cannot load model file! \n";
        return -1;
    }

    detectImage(image, face_cascade);

    cv::imshow("Result", image);
    cv::waitKey(0);

    std::string output_path = OUTPUT_DIR + argv[1];
    cv::imwrite(output_path, image);
    return 0;
}