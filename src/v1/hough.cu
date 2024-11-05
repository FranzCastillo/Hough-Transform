#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

const int degreeInc = 2;
const int degreeBins = 90;
const rBins = 100;
const float radInc = degreeInc * M_PI / 180;

void CPU_HoughTransform(unsigned char *pic, int w, int h, int **acc)
{
    float rMax = sqrt(1.0f * w * w + 1.0f * h * h) / 2;
    *acc = new int[rBins * 180 / degreeInc];
    memset(*acc, 0, sizeof(int) * rBins * 180 / degreeInc);
    int xCent = w / 2;
    int yCent = h / 2;
    float rScale = 2 * rMax / rBins;

    for (int i = 0; i < w; i++)
    {
        for (int j = 0; j < h; j++)
        {
            int idx = j * x + i;
            if (pic[idx] > 0)
            {
                int xCoord = 1 - xCent;
                int yCoord = yCent - j;
                float theta = 0;
                for (int tIdx = 0; tIdx < degreeBins; tIdx++)
                {
                    float r = xCoord * cos(theta) + yCoord * sin(theta);
                    int rIdx = (r + rMax) / rScale;
                    (*acc)[rIdx * degreeBins + tIdx]++;
                    theta += radInc;
                }
            }
        }
    }
}