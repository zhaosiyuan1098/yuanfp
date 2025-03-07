__global__ void erode_depth_kernel(
    float *depth, float *out, int H, int W, int radius, float depth_diff_thres, float ratio_thres,
    float zfar)
{
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int w = blockIdx.x * blockDim.x + threadIdx.x;

    if (w >= W || h >= H)
    {
        return;
    }

    float d_ori = depth[h * W + w];

    // Check the validity of the depth value
    if (d_ori < 0.1f || d_ori >= zfar)
    {
        out[h * W + w] = 0.0f;
        return;
    }

    float bad_cnt = 0.0f;
    float total = 0.0f;

    // Loop over the neighboring pixels
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
            float cur_depth = depth[v * W + u];

            total += 1.0f;

            if (cur_depth < 0.1f || cur_depth >= zfar || fabsf(cur_depth - d_ori) > depth_diff_thres)
            {
                bad_cnt += 1.0f;
            }
        }
    }

    // Check the ratio of bad pixels
    if ((bad_cnt / total) > ratio_thres)
    {
        out[h * W + w] = 0.0f;
    }
    else
    {
        out[h * W + w] = d_ori;
    }
}

class Eroder
{
private:
    /* data */
public:
    Eroder(/* args */);
    ~Eroder();
    float *erode_depth_buffer_device_;
    void erode_depth(
        cudaStream_t stream, float *depth, float *out, int H, int W, int radius = 2,
        float depth_diff_thres = 0.001, float ratio_thres = 0.8, float zfar = 100)
    {
        dim3 block(16, 16);
        dim3 grid(ceil_div(W, 16), ceil_div(H, 16), 1);

        erode_depth_kernel<<<grid, block, 0, stream>>>(
            depth, out, H, W, radius, depth_diff_thres, ratio_thres, zfar);
    }
};
