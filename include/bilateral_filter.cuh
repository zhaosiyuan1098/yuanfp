__global__ void bilateral_filter_depth_kernel(
    float *depth, float *out, int H, int W, float zfar, int radius, float sigmaD, float sigmaR)
{
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int w = blockIdx.x * blockDim.x + threadIdx.x;

    if (w >= W || h >= H)
    {
        return;
    }

    out[h * W + w] = 0.0f;

    // Compute the mean depth of the neighboring pixels
    float mean_depth = 0.0f;
    int num_valid = 0;
    for (int u = w - radius; u <= w + radius; u++)
    {
        if (u < 0 || u >= W)
        {
            continue;
        }
        for (int v = h - radius; v <= h + radius; v++)
        {
            if (v < 0 || v >= H)
            {
                continue;
            }
            // Get the current depth value
            float cur_depth = depth[v * W + u];
            if (cur_depth >= 0.1f && cur_depth < zfar)
            {
                num_valid++;
                mean_depth += cur_depth;
            }
        }
    }
}

class Bilateral_Filter
{
private:
    /* data */
public:
    Bilateral_Filter(/* args */);
    ~Bilateral_Filter();
    float *bilateral_depth_buffer_device_; // 声明变量
    std::vector<float> bilateral_depth_buffer_host_;
    
    void bilateral_filter_depth(
        cudaStream_t stream, float *depth, float *out, int H, int W, float zfar = 100, int radius = 2, float sigmaD = 2,
        float sigmaR = 100000)
    {
        dim3 block(16, 16);
        dim3 grid(ceil_div(W, 16), ceil_div(H, 16), 1);

        bilateral_filter_depth_kernel<<<grid, block, 0, stream>>>(depth, out, H, W, zfar, radius, sigmaD, sigmaR);
    }
};
