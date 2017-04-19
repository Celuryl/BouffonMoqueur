#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <dlib/opencv/cv_image.h>

using namespace dlib;
using namespace std;

// ----------------------------------------------------------------------------------------
static const char* LandmarksDataFile = "Resources\shape_predictor_68_face_landmarks.dat";

cv::Mat nextInput;

// ----------------------------------------------------------------------------------------

void webcam();

// ----------------------------------------------------------------------------------------

int main(int argc, char* argv[])
{
	//webcam();

	try
	{
		cv::VideoCapture cap;
		cap.open(0);
		if (!cap.isOpened())
		{
			cerr << "Unable to connect to camera" << endl;
			return 1;
		}

		image_window win;

		// Load face detection and pose estimation models.
		frontal_face_detector detector = get_frontal_face_detector();
		shape_predictor pose_model;
		deserialize("shape_predictor_68_face_landmarks.dat") >> pose_model;

		// Grab and process frames until the main window is closed by the user.
		while (!win.is_closed())
		{
			// Grab a frame
			cv::Mat temp;
			cap >> temp;
			cv_image<bgr_pixel> cimg(temp);

			// Detect faces 
			std::vector<rectangle> faces = detector(cimg);
			// Find the pose of each face.
			std::vector<full_object_detection> shapes;
			for (unsigned long i = 0; i < faces.size(); ++i)
				shapes.push_back(pose_model(cimg, faces[i]));

			// Display it all on the screen
			win.clear_overlay();
			win.set_image(cimg);
			win.add_overlay(render_face_detections(shapes));
		}
	}
	catch (serialization_error& e)
	{
		cout << "You need dlib's default face landmarking model file to run this example." << endl;
		cout << "You can get it from the following URL: " << endl;
		cout << "   http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" << endl;
		cout << endl << e.what() << endl;
	}
	catch (exception& e)
	{
		cout << e.what() << endl;
	}
}

void webcam()
{
	auto continue_bool = true;
	cv::VideoCapture cap;

	cap.open(0);
	if (!cap.isOpened()) throw std::exception("Failed to open video capture stream");

	while (continue_bool)
	{
		cap.read(nextInput);
		if (nextInput.empty()) throw std::exception("Failed to read from video capture stream");

		//Detect Face
		//Draw

		//Wait 10ms for user input, exit if any
		if (cv::waitKey(10) != -1) continue_bool = false;
	}
}