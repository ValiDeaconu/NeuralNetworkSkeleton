#include <cstdio>
#include <vector>
#include <cstring>
#include <cassert>
#include "../include/NeuralNetwork.hpp"

using namespace std;

struct inputData {
    unsigned int generations;
    vector<int> topology;
    vector<double> input;
    vector<double> target;
};

inputData fileReader(const char * path, int type);

int main(int argc, char ** argv) {
    switch (argc) {
        case 1:
            printf("USAGE: %s train|response\n", argv[0]);
            return 0;

        case 2:
            if (!strcmp(argv[1], "train")) {
                printf("USAGE: %s train new|old\n", argv[0]);
                return 0;
            } else if (!strcmp(argv[1], "response")) {
                printf("USAGE: %s response <path_to_input_file> <path_to_progress_file>\n", argv[0]);
                return 0;
            } else {
                printf("USAGE: %s train|response\n", argv[0]);
                return 0;
            }
        case 3:
            if (!strcmp(argv[1], "train")) {
                if (!strcmp(argv[2], "new") || !strcmp(argv[2], "old")) {
                    printf("USAGE: %s train %s <path_to_input_file> <path_to_progress_file>\n", argv[0], argv[2]);
                    return 0;
                } else {
                    printf("USAGE: %s train new|old\n", argv[0]);
                    return 0;
                }
            } else if (!strcmp(argv[1], "response")) {
                printf("USAGE: %s response <path_to_input_file> <path_to_progress_file>\n", argv[0]);
                return 0;
            } else {
                printf("USAGE: %s train|response\n", argv[0]);
                return 0;
            }
        case 4:
            if (!strcmp(argv[1], "train")) {
                printf("USAGE: %s train new|old\n", argv[0]);
                return 0;
            } else if (!strcmp(argv[1], "response")) {
                // argv[2] = path_to_input_file

                /** 
                 * INPUT FILE FORMAT 2
                 * <input_size>
                 * <input_vector>
                 * <target_size>
                 * <target_vector>
                 */
                inputData in = fileReader(argv[2], 2);

                NeuralNetwork *nn = new NeuralNetwork(argv[3]);
                nn->setInput(in.input);
                nn->setTarget(in.target);
                nn->feedForward();
                nn->printTarget();
                nn->printOutput();

                delete nn;

                break;
            } else {
                printf("USAGE: %s train|response\n", argv[0]);
                return 0;
            }
        case 5:
            if (!strcmp(argv[1], "train")) {
                if (!strcmp(argv[2], "new")) {
                    // argv[3] = path_to_input_file
                    // argv[4] = path_to_progres_file

                    /** 
                     * INPUT FILE FORMAT 0
                     * <generations>
                     * <topology_size>
                     * <topology_vector>
                     * <input_size>
                     * <input_vector>
                     * <target_size>
                     * <target_vector>
                     */

                    inputData in = fileReader(argv[3], 0);

                    NeuralNetwork *nn = new NeuralNetwork(in.topology);
                    nn->setInput(in.input);
                    nn->setTarget(in.target);

                    // Training process
                    unsigned int proc = (int)(0.10 * in.generations);
                    for (unsigned int g = 0; g < in.generations; ++g) {
                        nn->feedForward();
                        nn->setErrors();
                        if (g % proc == 0) {
                            printf("Generation: %d\n", g);
                            printf("Total error: %.6f\n", nn->getTotalError());
                        }
                        nn->backPropagation();
                    }

                    nn->saveProgress(argv[4]);
                    delete nn;

                    break;
                } else if (!strcmp(argv[2], "old")) {
                    // argv[3] = path_to_input_file
                    // argv[4] = path_to_progres_file

                    /** 
                     * INPUT FILE FORMAT 1
                     * <generations>
                     * <input_size>
                     * <input_vector>
                     * <target_size>
                     * <target_vector>
                     */

                    inputData in = fileReader(argv[3], 1);

                    NeuralNetwork *nn = new NeuralNetwork(argv[4]);
                    nn->setInput(in.input);
                    nn->setTarget(in.target);

                    // Training process
                    unsigned int proc = (int)(0.10 * in.generations);
                    for (unsigned int g = 0; g < in.generations; ++g) {
                        nn->feedForward();
                        nn->setErrors();
                        if (g % proc == 0) {
                            printf("Generation: %d\n", g);
                            printf("Total error: %.6f\n", nn->getTotalError());
                        }
                        nn->backPropagation();
                    }

                    nn->saveProgress(argv[4]);
                    delete nn;

                    break;
                } else {
                    printf("USAGE: %s train new|old\n", argv[0]);
                    return 0;
                }
            } else if (!strcmp(argv[1], "response")) {
                printf("USAGE: %s response <path_to_input_file> <path_to_progress_file>\n", argv[0]);
                return 0;
            } else {
                printf("USAGE: %s train|response\n", argv[0]);
                return 0;
            }

        default:
            printf("USAGE: %s train|response\n", argv[0]);
            return 0;
    }    
    return 0;
}

inputData fileReader(const char * path, int type) {
    /** 
     * INPUT FILE FORMAT 0
     * <generations>
     * <topology_size>
     * <topology_vector>
     * <input_size>
     * <input_vector>
     * <target_size>
     * <target_vector>
     */
    /** 
     * INPUT FILE FORMAT 1
     * <generations>
     * <input_size>
     * <input_vector>
     * <target_size>
     * <target_vector>
     */
    /** 
     * INPUT FILE FORMAT 2
     * <input_size>
     * <input_vector>
     * <target_size>
     * <target_vector>
     */

    FILE * fin = fopen(path, "r");
    assert(fin != NULL);

    inputData in;
    if (type == 0 || type == 1) {
        assert(fscanf(fin, "%u", &in.generations) == 1);
    }
    
    unsigned int inputSize = 0;
    unsigned int targetSize = 0;
    unsigned int topologySize = 0;
    double y = 0.0f;
    int x = 0;

    if (type == 0) {
        assert(fscanf(fin, "%u", &topologySize) == 1);
        for (unsigned int i = 0; i < topologySize; ++i) {
            assert(fscanf(fin, "%d", &x) == 1);
            in.topology.push_back(x);
        }
    }

    assert(fscanf(fin, "%u", &inputSize) == 1);
    for (unsigned int i = 0; i < inputSize; ++i) {
        assert(fscanf(fin, "%lf", &y) == 1);
        in.input.push_back(y);
    }

    assert(fscanf(fin, "%u", &targetSize) == 1);
    for (unsigned int i = 0; i < targetSize; ++i) {
        assert(fscanf(fin, "%lf", &y) == 1);
        in.target.push_back(y);
    }

    fclose(fin);

    return in;
}