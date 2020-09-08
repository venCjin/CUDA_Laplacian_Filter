#include <iostream>
#include "lodepng.h"
#include "kernel.h"

int main(int argc, char** argv) {
    if(argc != 3) {
        std::cout << "Run with input and output image filenames." << std::endl;
        return -1;
    }

    const char* input_file = "Img/input.png";// argv[1];
    const char* output_file = "Img/output3x3.png";// argv[2];

    std::vector<unsigned char> in_image;
    unsigned int width, height, error;

    error = lodepng::decode(in_image, width, height, input_file, LCT_RGB);
    if (error) std::cout << "decoder error " << error << ": " << lodepng_error_text(error) << std::endl;

    unsigned char* output_image = new unsigned char[in_image.size()];

    filter(in_image.data(), output_image, width, height);

    error = lodepng::encode(output_file, output_image, width, height, LCT_RGB);
    if(error) std::cout << "encoder error " << error << ": "<< lodepng_error_text(error) << std::endl;

    delete[] output_image;
    return 0;
}
