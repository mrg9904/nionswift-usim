"""Rasterize independent prism channels in one CUDA launch."""
import numpy as np


class GPUParticleRasterizer:
    def __init__(self):
        import cupy as cp
        if not cp.cuda.runtime.getDeviceCount():
            raise RuntimeError('CUDA unavailable')
        self.cp = cp
        self.kernel = cp.RawKernel(r'''
            #include <math_constants.h>
            extern "C" __global__ void particles(const double* triangles,
                const double* x, const double* y, const int* rectangles,
                const int* offsets, const int* tasks, float* lower, float* upper) {
                int particle=tasks[2*blockIdx.x], start=tasks[2*blockIdx.x+1];
                const int* r=rectangles+4*particle;
                int width=r[1]-r[0], count=width*(r[3]-r[2]);
                for (int pixel=start+threadIdx.x; pixel<min(count,start+4096); pixel+=blockDim.x) {
                    double px=x[r[0]+pixel%width], py=y[r[2]+pixel/width];
                    double lo=CUDART_INF, hi=-CUDART_INF;
                    for (int face=0; face<20; ++face) {
                        const double* t=triangles+(particle*20+face)*9;
                        // Match the original per-triangle XY clipping exactly.
                        if (px<fmin(t[0],fmin(t[3],t[6])) || px>fmax(t[0],fmax(t[3],t[6])) ||
                            py<fmin(t[1],fmin(t[4],t[7])) || py>fmax(t[1],fmax(t[4],t[7]))) continue;
                        double ax=t[3]-t[0], ay=t[4]-t[1], az=t[5]-t[2];
                        double bx=t[6]-t[0], by=t[7]-t[1], bz=t[8]-t[2];
                        double det=ax*by-ay*bx;
                        if (det==0.) continue;
                        double inv=1./det, dx=px-t[0], dy=py-t[1];
                        double a=(dx*by-dy*bx)*inv, b=(dy*ax-dx*ay)*inv;
                        if (a>=-1e-8 && b>=-1e-8 && a+b<=1.+1e-8) {
                            double z=t[2]+a*az+b*bz;
                            lo=fmin(lo,z); hi=fmax(hi,z);
                        }
                    }
                    lower[offsets[particle]+pixel]=isfinite(lo) ? (float)lo : CUDART_NAN_F;
                    upper[offsets[particle]+pixel]=isfinite(hi) ? (float)hi : CUDART_NAN_F;
                }
            }
        ''', 'particles', options=('--fmad=false',))

    def project(self, triangles, rectangles, x, y):
        cp = self.cp
        counts = (rectangles[:, 1]-rectangles[:, 0])*(rectangles[:, 3]-rectangles[:, 2])
        offsets = np.concatenate(([0], np.cumsum(counts))).astype(np.int32)
        tasks = np.asarray([(i, start) for i, count in enumerate(counts) for start in range(0, int(count), 4096)], np.int32)
        lower, upper = cp.empty(int(offsets[-1]), cp.float32), cp.empty(int(offsets[-1]), cp.float32)
        self.kernel((len(tasks),), (256,), (cp.asarray(triangles), cp.asarray(x), cp.asarray(y),
            cp.asarray(rectangles), cp.asarray(offsets), cp.asarray(tasks), lower, upper))
        lower, upper = cp.asnumpy(lower), cp.asnumpy(upper)
        return [(lower[offsets[i]:offsets[i+1]].reshape(rectangles[i, 3]-rectangles[i, 2], rectangles[i, 1]-rectangles[i, 0]),
                 upper[offsets[i]:offsets[i+1]].reshape(rectangles[i, 3]-rectangles[i, 2], rectangles[i, 1]-rectangles[i, 0]))
                for i in range(len(counts))]
