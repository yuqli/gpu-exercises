//
// Created by YUQIONG LI on 13/10/2018.
//

/*
 * Purpose: This .c program solves an image classification problem by :
 * 1) reads in training set pictures,
 * 2) build a convolution layer to them,
 * 3) and predict results on the test set.
 *
 * To-do's in next release:
 * 1) add GPU kernel
 * 2) add cross validation code
 * 3) migrant to multiple GPU
 * 4) try other predictive analytics algorithms: SVM, XGboost, etc..
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <iterator>
#include <string>

#define index(i, j, N)  ((i) * (N)) + (j);

using std::vector;
using std::cout;
using std::endl;
using std::ifstream;
using std::ios;
using std::string;

class Image {
public:
    Image(unsigned char l, vector<unsigned char> r, vector<unsigned char> g, vector<unsigned char> b) {
        // constructor for storing labels and color
        _l = l;
        _r = r;
        _g = g;
        _b = b;
    }

private:
    unsigned char _l;
    vector<unsigned char> _r, _g, _b;

};

vector<Image *> readABatch(string fpath, int num_pics, int width, int height){
    int length = width * height;

    ifstream is;
    is.open (fpath, ios::binary | ios::in);

    vector<Image *> v; // to store all images

    if (!is){
        cout << "Error! File is not open.\n";
        return v;  // return an empty vector
    }

    std::istreambuf_iterator<char> eos;
    std::istreambuf_iterator<char> iter(is);  // iterator for stream
    while(iter != eos){
        // while iterator has not reached end of file
        unsigned char l;
        vector<unsigned char> r, g, b;
        l = *iter;  // read label
        iter++;
        // read red
        for (int i = 0; i < 1024; i++){
            r.push_back(*iter);
            iter++;
        }
        // read green
        for (int i = 0; i < 1024; i++){
            g.push_back(*iter);
            iter++;
        }
        // read blue
        for (int i = 0; i < 1024; i++){
            r.push_back(*iter);
            iter++;
        }

        Image * curr = new Image(l, r, g, b);
        v.push_back(curr);
    }

    cout << "Finished reading " + std::to_string(v.size()) + " pictures!\n";
    is.close();
    return v;
}

int main () {
    vector<Image *> v;
    string fpath = "/Users/yuqiongli/desktop/gpu/project/cifar-10-batches-bin/data_batch_1.bin";
    int num_pics = 10000;
    int width = 32;
    int height = 32;

    clock_t start, end;
    double time_taken;
    start = clock();

    v = readABatch(fpath, num_pics, width, height);

    end = clock();
    time_taken = ((double)(end - start))/ CLOCKS_PER_SEC;
    printf("Reading files taken %.2lf seconds.", time_taken);

    /*
     * First convolution layer
     */
    int num_kernels1 = 10;
    int k_width = 5;
    int k_heigth = 5;
    int stride = 3;

    
    return 0;
}
