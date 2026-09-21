
#define _CRT_SECURE_NO_WARNINGS
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <vector>
#include <string>
#include <array>
#include <algorithm>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <functional>
#include <random>

#ifdef USE_STB_IMAGE
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#endif

// =====================================================================================
//  Basic math
// =====================================================================================
static const float PI_F = 3.14159265358979f;
static const float C_LIGHT = 10.0f;   // simulation units: G = 1, c = 10, rs = 2GM/c^2

struct V3 { float x, y, z; };
static inline V3 operator+(V3 a, V3 b) { return { a.x + b.x, a.y + b.y, a.z + b.z }; }
static inline V3 operator-(V3 a, V3 b) { return { a.x - b.x, a.y - b.y, a.z - b.z }; }
static inline V3 operator*(V3 a, float s) { return { a.x * s, a.y * s, a.z * s }; }
static inline float dot3(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static inline V3 cross3(V3 a, V3 b) { return { a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x }; }
static inline float len3(V3 a) { return sqrtf(dot3(a, a)); }
static inline V3 norm3(V3 a) { float l = len3(a); return l > 1e-12f ? a * (1.0f / l) : V3{ 0, 0, 0 }; }
static inline float clampf(float v, float a, float b) { return v < a ? a : (v > b ? b : v); }
static inline float sstep(float t) { t = clampf(t, 0.f, 1.f); return t * t * (3.f - 2.f * t); }

static std::mt19937 g_rng(1234567u);
static float rnd() { return (g_rng() & 0xFFFFFF) / 16777216.0f; }
static float rndR(float a, float b) { return a + (b - a) * rnd(); }
static float gauss() { static std::normal_distribution<float> nd(0.f, 1.f); return nd(g_rng); }

static void mat4Perspective(float* m, float fovy, float aspect, float zn, float zf) {
    float f = 1.0f / tanf(fovy * 0.5f);
    for (int i = 0; i < 16; i++) m[i] = 0.f;
    m[0] = f / aspect; m[5] = f;
    m[10] = (zf + zn) / (zn - zf); m[11] = -1.f;
    m[14] = 2.f * zf * zn / (zn - zf);
}
static void mat4LookAt(float* m, V3 eye, V3 center, V3 up) {
    V3 f = norm3(center - eye);
    V3 s = norm3(cross3(f, up));
    V3 u = cross3(s, f);
    m[0] = s.x; m[4] = s.y; m[8] = s.z; m[12] = -dot3(s, eye);
    m[1] = u.x; m[5] = u.y; m[9] = u.z; m[13] = -dot3(u, eye);
    m[2] = -f.x; m[6] = -f.y; m[10] = -f.z; m[14] = dot3(f, eye);
    m[3] = 0; m[7] = 0; m[11] = 0; m[15] = 1;
}

// =====================================================================================
//  Small persistent thread pool (main thread also works)
// =====================================================================================
class ThreadPool {
public:
    explicit ThreadPool(int n) { for (int i = 0; i < n; i++) workers.emplace_back([this] { loop(); }); }
    ~ThreadPool() {
        { std::lock_guard<std::mutex> lk(m); quit = true; }
        cvWork.notify_all();
        for (auto& t : workers) t.join();
    }
    void run(int count, const std::function<void(int)>& f) {
        if (count <= 0) return;
        if (workers.empty()) { for (int i = 0; i < count; i++) f(i); return; }
        {
            std::lock_guard<std::mutex> lk(m);
            task = &f; total = count; next.store(0); finished = 0; ++gen;
        }
        cvWork.notify_all();
        work();
        std::unique_lock<std::mutex> lk(m);
        cvDone.wait(lk, [&] { return finished == (int)workers.size(); });
    }
private:
    void work() { for (;;) { int i = next.fetch_add(1); if (i >= total) break; (*task)(i); } }
    void loop() {
        unsigned long long seen = 0;
        for (;;) {
            {
                std::unique_lock<std::mutex> lk(m);
                cvWork.wait(lk, [&] { return quit || gen != seen; });
                if (quit) return;
                seen = gen;
            }
            work();
            { std::lock_guard<std::mutex> lk(m); ++finished; }
            cvDone.notify_all();
        }
    }
    std::vector<std::thread> workers;
    std::mutex m;
    std::condition_variable cvWork, cvDone;
    const std::function<void(int)>* task = nullptr;
    int total = 0;
    std::atomic<int> next{ 0 };
    int finished = 0;
    unsigned long long gen = 0;
    bool quit = false;
};

// =====================================================================================
//  Noise (for procedural fallback textures)
// =====================================================================================
static inline uint32_t hu(uint32_t x) { x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16; return x; }
static inline float h3(int x, int y, int z) {
    uint32_t h = hu((uint32_t)x * 0x8da6b343U ^ hu((uint32_t)y * 0xd8163841U ^ hu((uint32_t)z * 0xcb1ab31fU)));
    return (h & 0xFFFFFF) / 16777216.0f;
}
static float vnoise(float x, float y, float z) {
    float fx = floorf(x), fy = floorf(y), fz = floorf(z);
    int xi = (int)fx, yi = (int)fy, zi = (int)fz;
    float tx = x - fx, ty = y - fy, tz = z - fz;
    tx = tx * tx * (3 - 2 * tx); ty = ty * ty * (3 - 2 * ty); tz = tz * tz * (3 - 2 * tz);
    float c000 = h3(xi, yi, zi), c100 = h3(xi + 1, yi, zi);
    float c010 = h3(xi, yi + 1, zi), c110 = h3(xi + 1, yi + 1, zi);
    float c001 = h3(xi, yi, zi + 1), c101 = h3(xi + 1, yi, zi + 1);
    float c011 = h3(xi, yi + 1, zi + 1), c111 = h3(xi + 1, yi + 1, zi + 1);
    float a = c000 + (c100 - c000) * tx, b = c010 + (c110 - c010) * tx;
    float c = c001 + (c101 - c001) * tx, d = c011 + (c111 - c011) * tx;
    float e = a + (b - a) * ty, f = c + (d - c) * ty;
    return e + (f - e) * tz;
}
static float fbm(float x, float y, float z, int oct) {
    float a = 0.5f, s = 0.f;
    for (int i = 0; i < oct; i++) { s += a * vnoise(x, y, z); x *= 2.02f; y *= 2.02f; z *= 2.02f; a *= 0.5f; }
    return s;
}

// =====================================================================================
//  Image loading (BMP / TGA / PPM built-in, JPG/PNG through optional stb_image)
// =====================================================================================
struct Image {
    int w = 0, h = 0; bool alpha = false; std::vector<uint8_t> px;
    bool ok() const { return w > 0 && h > 0 && !px.empty(); }
};
static bool fileExists(const char* p) { FILE* f = fopen(p, "rb"); if (f) { fclose(f); return true; } return false; }
static std::vector<uint8_t> readAll(const std::string& path) {
    std::vector<uint8_t> d; FILE* f = fopen(path.c_str(), "rb"); if (!f) return d;
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    if (n > 0) { d.resize((size_t)n); size_t r = fread(d.data(), 1, (size_t)n, f); if (r != (size_t)n) d.clear(); }
    fclose(f); return d;
}
static bool loadPPM(const std::vector<uint8_t>& d, Image& im) {
    if (d.size() < 10 || d[0] != 'P' || d[1] != '6') return false;
    size_t p = 2; int v[3];
    for (int k = 0; k < 3; k++) {
        while (p < d.size()) { if (d[p] == '#') { while (p < d.size() && d[p] != '\n') p++; } else if (d[p] == ' ' || d[p] == '\n' || d[p] == '\r' || d[p] == '\t') p++; else break; }
        int x = 0; while (p < d.size() && d[p] >= '0' && d[p] <= '9') { x = x * 10 + (d[p] - '0'); p++; }
        v[k] = x;
    }
    p++;
    int w = v[0], h = v[1];
    if (w <= 0 || h <= 0 || v[2] > 255 || p + (size_t)w * h * 3 > d.size()) return false;
    im.w = w; im.h = h; im.alpha = false; im.px.resize((size_t)w * h * 4);
    for (int i = 0; i < w * h; i++) { im.px[4 * i] = d[p + 3 * i]; im.px[4 * i + 1] = d[p + 3 * i + 1]; im.px[4 * i + 2] = d[p + 3 * i + 2]; im.px[4 * i + 3] = 255; }
    return true;
}
static bool loadBMP(const std::vector<uint8_t>& d, Image& im) {
    if (d.size() < 54 || d[0] != 'B' || d[1] != 'M') return false;
    auto rd32 = [&](size_t o) { return (uint32_t)d[o] | ((uint32_t)d[o + 1] << 8) | ((uint32_t)d[o + 2] << 16) | ((uint32_t)d[o + 3] << 24); };
    uint32_t off = rd32(10); int32_t w = (int32_t)rd32(18), h = (int32_t)rd32(22);
    int bpp = d[28] | (d[29] << 8); uint32_t comp = rd32(30);
    if ((bpp != 24 && bpp != 32) || (comp != 0 && comp != 3)) return false;
    bool topdown = h < 0; if (topdown) h = -h;
    if (w <= 0 || h <= 0) return false;
    int bytes = bpp / 8; size_t stride = (((size_t)w * bpp + 31) / 32) * 4;
    if (off + stride * (size_t)h > d.size()) return false;
    im.w = w; im.h = h; im.alpha = false; im.px.resize((size_t)w * h * 4);
    for (int y = 0; y < h; y++) {
        int sr = topdown ? y : (h - 1 - y);
        const uint8_t* s = &d[off + (size_t)sr * stride];
        for (int x = 0; x < w; x++) {
            uint8_t* o = &im.px[((size_t)y * w + x) * 4];
            o[0] = s[x * bytes + 2]; o[1] = s[x * bytes + 1]; o[2] = s[x * bytes]; o[3] = 255;
        }
    }
    return true;
}
static bool loadTGA(const std::vector<uint8_t>& d, Image& im) {
    if (d.size() < 18) return false;
    int idlen = d[0], type = d[2], w = d[12] | (d[13] << 8), h = d[14] | (d[15] << 8), bpp = d[16], desc = d[17];
    if (type != 2 || (bpp != 24 && bpp != 32) || w <= 0 || h <= 0) return false;
    size_t off = 18 + idlen; int bytes = bpp / 8;
    if (off + (size_t)w * h * bytes > d.size()) return false;
    bool top = (desc & 0x20) != 0;
    im.w = w; im.h = h; im.alpha = (bpp == 32); im.px.resize((size_t)w * h * 4);
    for (int y = 0; y < h; y++) {
        int sr = top ? y : (h - 1 - y);
        const uint8_t* s = &d[off + (size_t)sr * w * bytes];
        for (int x = 0; x < w; x++) {
            uint8_t* o = &im.px[((size_t)y * w + x) * 4];
            o[0] = s[x * bytes + 2]; o[1] = s[x * bytes + 1]; o[2] = s[x * bytes]; o[3] = bytes == 4 ? s[x * bytes + 3] : 255;
        }
    }
    return true;
}
static bool loadImageFile(const std::string& path, Image& im) {
#ifdef USE_STB_IMAGE
    {
        int w, h, n; unsigned char* p = stbi_load(path.c_str(), &w, &h, &n, 4);
        if (p) { im.w = w; im.h = h; im.alpha = (n == 4); im.px.assign(p, p + (size_t)w * h * 4); stbi_image_free(p); return true; }
    }
#endif
    std::string ext;
    size_t dp = path.find_last_of('.');
    if (dp != std::string::npos) { ext = path.substr(dp); for (auto& c : ext) c = (char)tolower(c); }
    std::vector<uint8_t> d = readAll(path);
    if (d.empty()) return false;
    if (ext == ".ppm") return loadPPM(d, im);
    if (ext == ".bmp") return loadBMP(d, im);
    if (ext == ".tga") return loadTGA(d, im);
    return false;
}
static void halveImage(Image& im) {
    int nw = im.w / 2, nh = im.h / 2; if (nw < 1 || nh < 1) return;
    std::vector<uint8_t> o((size_t)nw * nh * 4);
    for (int y = 0; y < nh; y++) for (int x = 0; x < nw; x++) for (int c = 0; c < 4; c++) {
        int s = im.px[((size_t)(2 * y) * im.w + 2 * x) * 4 + c] + im.px[((size_t)(2 * y) * im.w + 2 * x + 1) * 4 + c]
            + im.px[((size_t)(2 * y + 1) * im.w + 2 * x) * 4 + c] + im.px[((size_t)(2 * y + 1) * im.w + 2 * x + 1) * 4 + c];
        o[((size_t)y * nw + x) * 4 + c] = (uint8_t)(s / 4);
    }
    im.w = nw; im.h = nh; im.px.swap(o);
}
static void flipRows(Image& im) {
    size_t row = (size_t)im.w * 4;
    for (int y = 0; y < im.h / 2; y++) std::swap_ranges(im.px.begin() + y * row, im.px.begin() + (y + 1) * row, im.px.begin() + (size_t)(im.h - 1 - y) * row);
}

// =====================================================================================
//  Procedural fallback textures (used when no image files are found)
// =====================================================================================
static void genSky(std::vector<uint8_t>& out, int W, int H, ThreadPool& pool) {
    out.assign((size_t)W * H * 4, 255);
    pool.run(H, [&](int y) {
        float lat = ((y + 0.5f) / H - 0.5f) * PI_F, cl = cosf(lat), sl = sinf(lat);
        for (int x = 0; x < W; x++) {
            float lon = ((x + 0.5f) / W - 0.5f) * 2.f * PI_F;
            float dx = cl * cosf(lon), dy = sl, dz = cl * sinf(lon);
            float sh = dx * 0.30f + dy * 0.93f + dz * 0.21f;
            float band = expf(-(sh * sh) / (2.f * 0.16f * 0.16f));
            float f1 = fbm(dx * 3.f + 1.7f, dy * 3.f + 9.2f, dz * 3.f + 4.1f, 5);
            float f2 = fbm(dx * 6.5f + 11.f, dy * 6.5f + 3.f, dz * 6.5f + 7.f, 4);
            float f3 = fbm(dx * 1.8f + 30.f, dy * 1.8f + 20.f, dz * 1.8f + 10.f, 4);
            float dust = sstep((f2 - 0.42f) * 4.f);
            float core = band * (0.25f + 0.9f * f1) * (1.f - 0.65f * dust * band);
            float neb = powf(clampf((f3 - 0.45f) * 2.2f, 0.f, 1.f), 2.f);
            float r = core * 0.55f + neb * 0.30f * (0.6f + 0.4f * band) + band * band * f3 * 0.10f;
            float g = core * 0.50f + neb * 0.10f;
            float b = core * 0.62f + neb * 0.42f;
            uint8_t* o = &out[((size_t)y * W + x) * 4];
            o[0] = (uint8_t)(clampf(r, 0.f, 1.f) * 255.f); o[1] = (uint8_t)(clampf(g, 0.f, 1.f) * 255.f);
            o[2] = (uint8_t)(clampf(b, 0.f, 1.f) * 255.f); o[3] = 255;
        }
        });
}
// inner (blue-white, relativistic) -> outer (deep red)
static void diskRamp(float k, float& r, float& g, float& b) {
    static const float S[5][3] = { { 0.82f, 0.90f, 1.00f },{ 1.00f, 0.96f, 0.86f },{ 1.00f, 0.74f, 0.34f },
                                  { 0.94f, 0.38f, 0.11f },{ 0.42f, 0.10f, 0.03f } };
    float u = clampf(k, 0.f, 1.f) * 4.f; int i = (int)u; if (i > 3) i = 3; float f = u - i;
    r = S[i][0] + (S[i + 1][0] - S[i][0]) * f;
    g = S[i][1] + (S[i + 1][1] - S[i][1]) * f;
    b = S[i][2] + (S[i + 1][2] - S[i][2]) * f;
}
// Polar strip: x = angle (wraps), y = radius (0 = inner edge, 1 = outer edge)
static void genDisk(std::vector<uint8_t>& out, int W, int H, ThreadPool& pool) {
    out.assign((size_t)W * H * 4, 0);
    pool.run(H, [&](int y) {
        float t = (y + 0.5f) / H, rn = 3.f + t * 14.f;
        float I = powf(3.f / rn, 2.15f);
        float fin = sstep(t / 0.045f), fout = 1.f - sstep((t - 0.52f) / 0.48f);
        float cr, cg, cb; diskRamp(powf(t, 0.5f), cr, cg, cb);
        float shear = -7.5f * powf(rn / 3.f, -1.5f);       // Keplerian winding of the filaments
        for (int x = 0; x < W; x++) {
            float ang = (x + 0.5f) / W * 2.f * PI_F + shear;
            float ca = cosf(ang), sa = sinf(ang);
            float n1 = fbm(ca * 2.2f + 20.f, sa * 2.2f + 20.f, t * 26.f, 5);
            float n2 = fbm(ca * 7.0f + 50.f, sa * 7.0f + 50.f, t * 80.f, 4);
            float n3 = fbm(ca * 17.f + 80.f, sa * 17.f + 80.f, t * 170.f, 3);
            float fil = powf(clampf(1.f - fabsf(n1 - 0.5f) * 3.3f, 0.f, 1.f), 1.7f);   // ridged streams
            float dens = clampf(0.28f + 1.55f * fil + 0.55f * (n2 - 0.42f) + 0.22f * (n3 - 0.5f), 0.03f, 1.7f);
            float br = I * dens;
            float hot = clampf(br * 0.55f, 0.f, 1.f);                                  // hot spots go white
            float alpha = clampf(0.07f + dens * (0.35f + br * 1.9f), 0.f, 0.985f) * fin * fout;
            float e = clampf(br * 1.55f + 0.04f, 0.f, 1.f);
            float r = (cr + (1.f - cr) * hot) * e, g = (cg + (1.f - cg) * hot * 0.85f) * e, b = (cb + (1.f - cb) * hot * 0.7f) * e;
            uint8_t* o = &out[((size_t)y * W + x) * 4];
            o[0] = (uint8_t)(clampf(r, 0.f, 1.f) * 255.f); o[1] = (uint8_t)(clampf(g, 0.f, 1.f) * 255.f);
            o[2] = (uint8_t)(clampf(b, 0.f, 1.f) * 255.f); o[3] = (uint8_t)(alpha * 255.f);
        }
        });
}

// =====================================================================================
//  Physics: N-body particles with self gravity per body + Paczynski-Wiita black hole
// =====================================================================================
struct KindDef { const char* name; float mass, a; int n; float cr, cg, cb; float size, lum; bool solid; };
static const KindDef KINDS[3] = {
    { "STAR",   1.0f,  0.50f, 260, 1.00f, 0.82f, 0.45f, 0.50f, 0.16f, false },
    { "GIANT",  4.0f,  1.10f, 300, 0.55f, 0.72f, 1.00f, 1.00f, 0.13f, false },
    { "PLANET", 0.15f, 0.18f, 140, 0.50f, 0.50f, 0.50f, 0.12f, 1.00f, true } };
static const float DENS[3] = { 0.5f, 1.0f, 1.6f };

struct Body {
    int kind = 0; float mp = 0, a = 0, eps2 = 0;
    std::vector<float> x, y, z, vx, vy, vz, ax, ay, az, cr, cg, cb;
    V3 com{ 0, 0, 0 }; bool disrupted = false; float intact = 1.f; bool hasCom = false;
    int count() const { return (int)x.size(); }
    void push(V3 p, V3 v, float r, float g, float b) {
        x.push_back(p.x); y.push_back(p.y); z.push_back(p.z);
        vx.push_back(v.x); vy.push_back(v.y); vz.push_back(v.z);
        ax.push_back(0); ay.push_back(0); az.push_back(0);
        cr.push_back(r); cg.push_back(g); cb.push_back(b);
    }
    void removeAt(int i) {
        std::vector<float>* A[12] = { &x, &y, &z, &vx, &vy, &vz, &ax, &ay, &az, &cr, &cg, &cb };
        int l = (int)x.size() - 1;
        for (auto* v : A) { (*v)[i] = (*v)[l]; v->pop_back(); }
    }
};
struct Sim {
    double Mbh = 50.0;                  // G = 1
    std::vector<Body> bodies;
    float accreted = 0.f, flare = 0.f;
    float feedRaw = 0.f;                // mass swallowed since last update
    float feed = 0.f;                   // smoothed accretion rate -> disk brightness & thickness
    float diskMass = 0.f;               // debris parked in the disk annulus -> disk density
    float jet = 0.f;                    // gamma-ray jet strength
    bool dirty = false;
    float rs() const { return (float)(2.0 * Mbh / (C_LIGHT * C_LIGHT)); }
};

static void spawnBody(Sim& s, int kind, V3 pos, V3 vel, float dens) {
    const KindDef& K = KINDS[kind];
    Body B; B.kind = kind; B.a = K.a; B.eps2 = (0.3f * K.a) * (0.3f * K.a);
    int n = std::max(40, (int)(K.n * dens)); B.mp = K.mass / n;
    std::vector<V3> P(n), Vv(n);
    V3 mP{ 0, 0, 0 }, mV{ 0, 0, 0 };
    for (int i = 0; i < n; i++) {
        float r;
        do { float X = rndR(0.002f, 0.998f); r = K.a / sqrtf(powf(X, -2.0f / 3.0f) - 1.0f); } while (r > 5.0f * K.a);
        float cz = rndR(-1.f, 1.f), ph = rndR(0.f, 2.f * PI_F), sn = sqrtf(1.f - cz * cz);
        P[i] = { r * sn * cosf(ph), r * sn * sinf(ph), r * cz };
        float sig = 0.95f * sqrtf(K.mass / (6.0f * sqrtf(r * r + K.a * K.a)));   // Plummer velocity dispersion
        Vv[i] = { gauss() * sig, gauss() * sig, gauss() * sig };
        mP = mP + P[i]; mV = mV + Vv[i];
    }
    mP = mP * (1.f / n); mV = mV * (1.f / n);
    for (int i = 0; i < n; i++) {
        V3 p = P[i] - mP + pos, v = Vv[i] - mV + vel;
        float cr, cg, cb;
        if (K.solid) {
            float t = rnd();
            if (t < 0.55f) { cr = 0.20f; cg = 0.40f; cb = 0.85f; }   // ocean
            else if (t < 0.88f) { cr = 0.55f; cg = 0.42f; cb = 0.28f; }   // land
            else { cr = 0.90f; cg = 0.90f; cb = 0.95f; }   // clouds
            float k = rndR(0.8f, 1.1f); cr *= k; cg *= k; cb *= k;
        }
        else {
            float k = rndR(0.85f, 1.1f); cr = K.cr * k; cg = K.cg * k; cb = K.cb * k;
        }
        B.push(p, v, cr, cg, cb);
    }
    if (s.bodies.size() >= 12) s.bodies.erase(s.bodies.begin());
    s.bodies.push_back(std::move(B));
    s.dirty = true;
}

static void computeAcc(Sim& s, ThreadPool& pool) {
    static std::vector<std::array<int, 3>> tasks;
    tasks.clear();
    long long work = 0;
    for (int b = 0; b < (int)s.bodies.size(); b++) {
        int n = s.bodies[b].count(); work += (long long)n * n;
        for (int lo = 0; lo < n; lo += 48) tasks.push_back({ b, lo, std::min(n, lo + 48) });
    }
    const float GM = (float)s.Mbh, rs = s.rs();
    const std::function<void(int)> fn = [&](int t) {
        Body& B = s.bodies[tasks[t][0]];
        const int lo = tasks[t][1], hi = tasks[t][2], n = B.count();
        const float mp = B.mp, e2 = B.eps2;
        const float* X = B.x.data(); const float* Y = B.y.data(); const float* Z = B.z.data();
        for (int i = lo; i < hi; i++) {
            float xi = X[i], yi = Y[i], zi = Z[i], sx = 0.f, sy = 0.f, sz = 0.f;
            for (int j = 0; j < n; j++) {
                float dx = X[j] - xi, dy = Y[j] - yi, dz = Z[j] - zi;
                float d2 = dx * dx + dy * dy + dz * dz + e2;
                float inv = 1.0f / sqrtf(d2), f = inv * inv * inv;
                sx += dx * f; sy += dy * f; sz += dz * f;
            }
            sx *= mp; sy *= mp; sz *= mp;
            float r = sqrtf(xi * xi + yi * yi + zi * zi);
            float d = r - rs; if (d < 0.05f) d = 0.05f;
            float g = -GM / (d * d * std::max(r, 1e-4f));      // Paczynski-Wiita: a = -GM/(r-rs)^2
            sx += g * xi; sy += g * yi; sz += g * zi;
            if (B.disrupted) {                                  // gas-like drag for debris close to the hole
                float w = clampf((10.f * rs - r) / (10.f * rs), 0.f, 1.f);
                float k = 0.35f * w * w;
                sx -= k * B.vx[i]; sy -= k * B.vy[i]; sz -= k * B.vz[i];
            }
            B.ax[i] = sx; B.ay[i] = sy; B.az[i] = sz;
        }
        };
    if (work < 60000) { for (int t = 0; t < (int)tasks.size(); t++) fn(t); }
    else pool.run((int)tasks.size(), fn);
}

static void stepSim(Sim& s, float h, ThreadPool& pool) {
    for (auto& B : s.bodies) {
        int n = B.count();
        for (int i = 0; i < n; i++) {
            B.vx[i] += 0.5f * h * B.ax[i]; B.vy[i] += 0.5f * h * B.ay[i]; B.vz[i] += 0.5f * h * B.az[i];
            B.x[i] += h * B.vx[i]; B.y[i] += h * B.vy[i]; B.z[i] += h * B.vz[i];
        }
    }
    computeAcc(s, pool);
    const float lim2 = (0.9f * C_LIGHT) * (0.9f * C_LIGHT);
    for (auto& B : s.bodies) {
        int n = B.count();
        for (int i = 0; i < n; i++) {
            B.vx[i] += 0.5f * h * B.ax[i]; B.vy[i] += 0.5f * h * B.ay[i]; B.vz[i] += 0.5f * h * B.az[i];
            float v2 = B.vx[i] * B.vx[i] + B.vy[i] * B.vy[i] + B.vz[i] * B.vz[i];
            if (v2 > lim2) { float k = sqrtf(lim2 / v2); B.vx[i] *= k; B.vy[i] *= k; B.vz[i] *= k; }
        }
    }
    // accretion into the black hole / escape to infinity
    for (auto& B : s.bodies) {
        for (int i = B.count() - 1; i >= 0; i--) {
            float rs = s.rs();
            float r = sqrtf(B.x[i] * B.x[i] + B.y[i] * B.y[i] + B.z[i] * B.z[i]);
            if (r < rs * 1.06f) { s.Mbh += B.mp; s.accreted += B.mp; s.flare += 0.8f * B.mp; s.feedRaw += B.mp; B.removeAt(i); }
            else if (r > 3000.f) B.removeAt(i);
        }
    }
    s.bodies.erase(std::remove_if(s.bodies.begin(), s.bodies.end(), [](const Body& b) { return b.count() == 0; }), s.bodies.end());
    if (s.flare > 3.f) s.flare = 3.f;
}

static void updatePhysics(Sim& s, float dtSim, ThreadPool& pool) {
    if (s.bodies.empty()) return;
    if (s.dirty) { computeAcc(s, pool); s.dirty = false; }
    int steps = (int)ceilf(dtSim / 0.004f); steps = std::max(1, std::min(steps, 16));
    float h = dtSim / steps;
    for (int i = 0; i < steps && !s.bodies.empty(); i++) stepSim(s, h, pool);
}
// Accretion rate drives disk brightness/thickness; falling matter launches the gamma-ray jet.
static void updateFeed(Sim& s, float dt) {
    dt = std::max(dt, 1e-4f);
    float rate = s.feedRaw / dt;
    s.feedRaw = 0.f;
    s.feed += (rate - s.feed) * (1.f - expf(-dt * 2.2f));
    if (s.feed < 1e-5f) s.feed = 0.f;

    float rs = s.rs(), m = 0.f;
    for (auto& B : s.bodies) {
        int n = B.count();
        for (int i = 0; i < n; i++) {
            float rr = sqrtf(B.x[i] * B.x[i] + B.z[i] * B.z[i]);
            if (rr > 2.f * rs && rr < 17.f * rs && fabsf(B.y[i]) < 3.f * rs) m += B.mp;
        }
    }
    s.diskMass += (m - s.diskMass) * (1.f - expf(-dt * 1.5f));

    float target = clampf(s.feed * 0.55f, 0.f, 2.6f);
    if (target > s.jet) s.jet += (target - s.jet) * (1.f - expf(-dt * 7.f));   // fast launch
    else s.jet *= expf(-dt * 0.40f);                                           // slow fade
    if (s.jet < 1e-4f) s.jet = 0.f;
}
static void updateStats(Sim& s) {
    for (auto& B : s.bodies) {
        int n = B.count(); if (n == 0) continue;
        V3 c{ 0, 0, 0 };
        if (B.hasCom) c = B.com;
        else { for (int i = 0; i < n; i++) c = c + V3{ B.x[i], B.y[i], B.z[i] }; c = c * (1.f / n); }
        for (int it = 0; it < 2; it++) {
            V3 sum{ 0, 0, 0 }; int cnt = 0; float lim = 3.f * B.a;
            for (int i = 0; i < n; i++) {
                V3 d = V3{ B.x[i], B.y[i], B.z[i] } - c;
                if (dot3(d, d) < lim * lim) { sum = sum + V3{ B.x[i], B.y[i], B.z[i] }; cnt++; }
            }
            if (cnt > 0) c = sum * (1.f / cnt);
        }
        B.com = c; B.hasCom = true;
        int far = 0; float lim = 6.f * B.a;
        for (int i = 0; i < n; i++) { V3 d = V3{ B.x[i], B.y[i], B.z[i] } - c; if (dot3(d, d) > lim * lim) far++; }
        float unb = (float)far / n;
        if (unb > 0.2f) B.disrupted = true;
        B.intact = 1.f - clampf(unb * 4.f, 0.f, 1.f);
    }
}

struct Vtx { float x, y, z, vx, vy, vz, r, g, b, size, heat, lum; };
static void buildVerts(Sim& s, std::vector<Vtx>& solid, std::vector<Vtx>& glow) {
    solid.clear(); glow.clear();
    float rs = s.rs();
    for (auto& B : s.bodies) {
        const KindDef& K = KINDS[B.kind]; int n = B.count();
        for (int i = 0; i < n; i++) {
            float r = sqrtf(B.x[i] * B.x[i] + B.y[i] * B.y[i] + B.z[i] * B.z[i]);
            float red = sqrtf(std::max(0.f, 1.f - rs / std::max(r, rs * 1.001f)));
            float prox = clampf(1.f - (r - rs) / (14.f * rs), 0.f, 1.f); prox *= prox;
            float sp = sqrtf(B.vx[i] * B.vx[i] + B.vy[i] * B.vy[i] + B.vz[i] * B.vz[i]) / C_LIGHT;
            float shock = clampf((sp - 0.10f) / 0.45f, 0.f, 1.f);
            float heat = clampf(0.55f * prox + 0.85f * shock * shock, 0.f, 1.f);
            Vtx v{ B.x[i], B.y[i], B.z[i], B.vx[i], B.vy[i], B.vz[i], B.cr[i], B.cg[i], B.cb[i], K.size, heat, 0.f };
            if (K.solid) { v.lum = 0.15f + 0.85f * red; solid.push_back(v); }
            else { v.lum = K.lum * (0.30f + 0.70f * red) * (1.f + 2.4f * heat); glow.push_back(v); }
        }
        if (!K.solid && B.intact > 0.02f) {                        // layered photosphere halo for the intact star
            float r = len3(B.com);
            float red = sqrtf(std::max(0.f, 1.f - rs / std::max(r, rs * 1.001f)));
            const float SZ[3] = { 3.2f, 6.5f, 12.0f }, LM[3] = { 0.55f, 0.20f, 0.07f };
            for (int k = 0; k < 3; k++)
                glow.push_back({ B.com.x, B.com.y, B.com.z, 0.f, 0.f, 0.f, K.cr, K.cg, K.cb,
                                B.a * SZ[k], 0.f, LM[k] * B.intact * red });
        }
    }
}

// =====================================================================================
//  Shaders
// =====================================================================================
static const char* VS_FULL = R"GLSL(#version 330 core
out vec2 vUV;
void main(){
    vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
    vUV = p;
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
)GLSL";

static const char* FS_BLIT = R"GLSL(#version 330 core
in vec2 vUV;
out vec4 c;
uniform sampler2D uTex;
void main(){ c = vec4(texture(uTex, vUV).rgb, 1.0); }
)GLSL";

// Black hole: approximate Schwarzschild null geodesics (a = -1.5 rs h^2 x / r^5), image-based sky and disk
// Black hole: approximate Schwarzschild null geodesics, image-based sky and disk, gamma-ray jets
static const char* FS_LENS = R"GLSL(#version 330 core
in vec2 vUV;
out vec4 fragColor;
uniform vec3 uCam; uniform vec3 uRight; uniform vec3 uUp; uniform vec3 uFwd;
uniform float uTanH; uniform float uAspect; uniform float uRs; uniform float uTime;
uniform float uFlare; uniform float uSkyLod; uniform float uDiskGain;
uniform float uFeed; uniform float uMassN; uniform float uJet;
uniform int uProcStars; uniform int uDiskOn; uniform int uBeam;
uniform sampler2D uSky; uniform sampler2D uDisk;
const float PI = 3.14159265359;
const int MAXSTEPS = 150;
const float DISK_SPEED = 3.0;

float hash13(vec3 p){ p = fract(p * 0.1031); p += dot(p, p.zyx + 31.32); return fract((p.x + p.y) * p.z); }
vec3 hash33(vec3 p){ p = fract(p * vec3(0.1031, 0.1030, 0.0973)); p += dot(p, p.yxz + 33.33); return fract((p.xxy + p.yxx) * p.zyx); }

vec3 starLayer(vec3 d, float scale, float density, float size){
    vec3 p = d * scale; vec3 ip = floor(p); vec3 fp = p - ip;
    if (hash13(ip) > density) return vec3(0.0);
    vec3 o = hash33(ip + 7.7) * 0.6 + 0.2;
    vec3 df = fp - o;
    float b = exp(-dot(df, df) / (size * size));
    float k = hash13(ip + 3.1);
    vec3 col = mix(vec3(1.0, 0.72, 0.5), vec3(0.62, 0.78, 1.0), k);
    float m = hash13(ip + 9.2);
    return col * b * (0.35 + 2.0 * m * m * m);
}
vec3 skyColor(vec3 d){
    float u = atan(d.z, d.x) / (2.0 * PI) + 0.5;
    float v = asin(clamp(d.y, -1.0, 1.0)) / PI + 0.5;
    vec3 c = textureLod(uSky, vec2(u, v), uSkyLod).rgb;
    if (uProcStars == 1) c += starLayer(d, 55.0, 0.10, 0.07) * 1.4 + starLayer(d, 130.0, 0.16, 0.09) * 0.7;
    return c;
}
// Collimated gamma-ray jet along the spin axis, fed by matter crossing the horizon
vec3 jetEmit(vec3 p){
    float ay = abs(p.y);
    if (ay < 0.85 * uRs) return vec3(0.0);
    float rho = length(p.xz);
    float rc = uRs * (0.20 + 0.14 * ay / uRs);
    float x = rho / rc;
    float core = exp(-x * x);
    float sheath = exp(-x * 0.65) * 0.22;
    float fall = exp(-ay / (30.0 * uRs));
    float ph = ay / uRs - uTime * 11.0;
    float knot = 0.62 + 0.38 * sin(ph * 1.7) * sin(ph * 0.41 + 1.3);
    float base = exp(-(ay - uRs) / (2.2 * uRs)) * 0.8;
    vec3 col = mix(vec3(0.58, 0.72, 1.0), vec3(0.98, 0.92, 1.0), core);
    return col * (core * knot + sheath + base * core) * fall * uJet * 0.16;
}
vec4 diskSample(vec3 p, vec3 rayDir, float rin, float rout){
    float rr = length(p.xz);
    float t = (rr - rin) / (rout - rin);
    float ang = atan(p.z, p.x);
    float om = DISK_SPEED / pow(max(rr / uRs, 1.0), 1.5);
    const float CYC = 10.0;
    float p1 = fract(uTime / CYC), p2 = fract(uTime / CYC + 0.5);
    float w1 = 1.0 - abs(2.0 * p1 - 1.0), w2 = 1.0 - abs(2.0 * p2 - 1.0);
    float lod = mix(0.0, 1.8, t);
    vec4 c1 = textureLod(uDisk, vec2((ang - om * p1 * CYC) / (2.0 * PI), t), lod);
    vec4 c2 = textureLod(uDisk, vec2((ang - om * p2 * CYC) / (2.0 * PI), t), lod);
    vec4 c = c1 * w1 + c2 * w2;
    float g = 1.0;
    if (uBeam == 1) {
        float rn = max(rr / uRs, 1.05);
        float beta = min(0.8, sqrt(0.5 * rn) / (rn - 1.0));
        vec3 tv = vec3(-p.z, 0.0, p.x) / max(rr, 1e-4);
        float cosT = dot(tv, -rayDir);
        float gam = inversesqrt(1.0 - beta * beta);
        float D = 1.0 / (gam * (1.0 - beta * cosT));
        float gr = sqrt(max(1.0 - 1.0 / rn, 0.0));
        g = clamp(D * gr, 0.25, 1.6);
    }
    // accretion rate + parked debris thicken, brighten and heat the disk
    float gain = uDiskGain * (0.55 + 1.7 * uFeed + 0.8 * uMassN) * (1.0 + uFlare);
    vec3 emit = c.rgb * gain * vec3(pow(g, 2.6), pow(g, 3.2), pow(g, 4.0));
    emit = mix(emit, vec3(dot(emit, vec3(0.33)) * 1.15) * vec3(0.85, 0.92, 1.1), clamp(uFeed * 0.45, 0.0, 0.6));
    float a = clamp(c.a * (0.85 + 0.85 * uMassN + 0.45 * uFeed), 0.0, 0.99);
    return vec4(emit, a);
}
void main(){
    vec2 ndc = vUV * 2.0 - 1.0;
    vec3 dir = normalize(uFwd + uRight * (ndc.x * uTanH * uAspect) + uUp * (ndc.y * uTanH));
    vec3 pos = uCam;
    float rs = uRs;
    vec3 hv = cross(pos, dir);
    float h2 = dot(hv, hv);
    float Resc = max(80.0 * rs, length(pos) * 1.02);
    float rin = 3.0 * rs, rout = 16.0 * rs;
    float hh = rs * 0.40 * (0.55 + 0.95 * uFeed + 0.5 * uMassN);
    vec3 col = vec3(0.0);
    float alpha = 0.0, rmin = 1e9;
    bool escaped = false;
    for (int i = 0; i < MAXSTEPS; i++) {
        float r = length(pos);
        rmin = min(rmin, r);
        if (r < rs) break;
        if (r > Resc && dot(pos, dir) > 0.0) { escaped = true; break; }
        float k = mix(0.035, 0.11, smoothstep(2.5, 10.0, r / rs));
        float dt = clamp(k * r, 0.02 * rs, 10.0);
        vec3 acc = -1.5 * rs * h2 * pos / (r * r * r * r * r);
        vec3 nd = normalize(dir + acc * dt);
        vec3 np = pos + nd * dt;
        if (uDiskOn == 1) {
            for (int L = -1; L <= 1; L++) {
                float yk = float(L) * hh;
                if ((pos.y - yk) * (np.y - yk) < 0.0) {
                    float tt = (pos.y - yk) / (pos.y - np.y);
                    vec3 pp = mix(pos, np, tt);
                    float rr = length(pp.xz);
                    if (rr > rin && rr < rout) {
                        float lw = (L == 0) ? 1.0 : 0.48;
                        vec4 e = diskSample(pp, nd, rin, rout);
                        float a = e.a * lw;
                        col += (1.0 - alpha) * a * e.rgb;
                        alpha += (1.0 - alpha) * a;
                    }
                }
            }
        }
        if (uJet > 0.002) col += (1.0 - alpha) * jetEmit(pos) * dt;
        pos = np; dir = nd;
        if (alpha > 0.985) break;
    }
    vec3 sky = vec3(0.0);
    if (escaped && alpha < 0.985) {
        sky = skyColor(normalize(dir));
        float g = max(rmin / rs - 1.5, 0.0);
        col += vec3(1.0, 0.82, 0.6) * (0.35 + 0.5 * uFeed) * exp(-g * 9.0) * (1.0 - alpha);
    }
    vec3 outc = (1.0 - exp(-col)) + sky * (1.0 - alpha);
    fragColor = vec4(clamp(outc, 0.0, 1.0), 1.0);
}
)GLSL";

// Particles: point sprites with a point-lens approximation (primary + secondary image)
// Velocity-stretched billboards + point-lens approximation (primary + secondary image)
    static const char* VS_PART = R"GLSL(#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aVel;
layout(location=2) in vec3 aCol;
layout(location=3) in float aSize;
layout(location=4) in float aHeat;
layout(location=5) in float aLum;
uniform mat4 uView; uniform mat4 uProj;
uniform vec3 uCam; uniform float uRs; uniform float uPx; uniform float uTrail;
uniform int uLens; uniform int uMode;
out vec2 vQ; out vec3 vCol; out float vHeat; out float vLum; out vec3 vLight;
void main(){
    vec2 q = vec2(((gl_VertexID & 1) == 0) ? -1.0 : 1.0, (gl_VertexID < 2) ? -1.0 : 1.0);
    vQ = q;
    vec3 P = aPos;
    vec3 s = P - uCam;
    float DS = max(length(s), 1e-3);
    float DL = max(length(uCam), 1e-3);
    vec3 axv = -uCam / DL;
    float zs = dot(s, axv);
    float DLS = zs - DL;
    float mag = 1.0;
    bool hide = false;
    vec3 Pimg = P;
    int img = gl_InstanceID & 1;
    if (uLens == 1 && DLS > 0.0 && zs > 0.0) {
        float beta = acos(clamp(zs / DS, -1.0, 1.0));
        vec3 e = s - zs * axv;
        float el = length(e);
        if (el > 1e-5) e /= el;
        else { vec3 t = abs(axv.y) < 0.99 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0); e = normalize(cross(axv, t)); }
        float th2 = 2.0 * uRs * DLS / (DL * zs);
        float thE = sqrt(th2);
        float sq = sqrt(beta * beta + 4.0 * th2);
        float th = (img == 0) ? 0.5 * (beta + sq) : 0.5 * (beta - sq);
        float u = max(beta / thE, 1e-3);
        float f = (u * u + 2.0) / (2.0 * u * sqrt(u * u + 4.0));
        mag = clamp((img == 0) ? f + 0.5 : f - 0.5, 0.05, 9.0);
        vec3 dimg = cos(th) * axv + sin(th) * e;
        Pimg = uCam + dimg * DS;
        float b = DL * tan(min(abs(th), 1.4));
        if (b < 2.6 * uRs) hide = true;
    } else if (img == 1) {
        hide = true;
    }
    vec4 vp = uView * vec4(Pimg, 1.0);
    float depth = max(-vp.z, 0.05);
    float w = aSize * 0.5 * sqrt(mag);
    w = max(w, depth * 2.0 / uPx);
    vec2 off;
    if (uMode == 1) {
        off = q * w;
    } else {
        vec3 vv = mat3(uView) * aVel;
        float sl = length(vv.xy);
        vec2 dir = (sl > 1e-4) ? vv.xy / sl : vec2(1.0, 0.0);
        float elong = min(w + sl * uTrail * 0.5, w * 9.0);
        off = vec2(-dir.y, dir.x) * (q.x * w) + dir * (q.y * elong);
    }
    vp.xy += off;
    gl_Position = uProj * vp;
    if (hide) gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
    vCol = aCol; vHeat = aHeat; vLum = aLum;
    vLight = normalize(mat3(uView) * (-P));
}
)GLSL";

    static const char* FS_PART = R"GLSL(#version 330 core
in vec2 vQ; in vec3 vCol; in float vHeat; in float vLum; in vec3 vLight;
uniform int uMode;
out vec4 frag;
void main(){
    float d2 = dot(vQ, vQ);
    if (d2 > 1.0) discard;
    vec3 c = vCol;
    c = mix(c, vec3(1.00, 0.55, 0.15), smoothstep(0.05, 0.40, vHeat));
    c = mix(c, vec3(1.00, 0.96, 0.88), smoothstep(0.40, 0.75, vHeat));
    c = mix(c, vec3(0.78, 0.86, 1.00), smoothstep(0.75, 1.00, vHeat));
    if (uMode == 0) {
        float g = exp(-d2 * 3.2);
        g = g * g * 0.65 + g * 0.70;
        frag = vec4(c * g * vLum, 1.0);
    } else {
        vec3 n = vec3(vQ.x, -vQ.y, sqrt(max(1.0 - d2, 0.0)));
        float diff = max(dot(n, normalize(vLight)), 0.0);
        vec3 o = c * (0.05 + 0.95 * diff) + vec3(1.0, 0.5, 0.15) * vHeat * 1.6;
        frag = vec4(o * vLum, 1.0);
    }
}
)GLSL";

static GLuint compileShader(GLenum type, const char* src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr); glCompileShader(s);
    GLint ok = 0; glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) { char log[4096]; glGetShaderInfoLog(s, 4096, nullptr, log); fprintf(stderr, "Shader compile error:\n%s\n", log); }
    return s;
}
static GLuint makeProgram(const char* vs, const char* fs) {
    GLuint v = compileShader(GL_VERTEX_SHADER, vs), f = compileShader(GL_FRAGMENT_SHADER, fs);
    GLuint p = glCreateProgram(); glAttachShader(p, v); glAttachShader(p, f); glLinkProgram(p);
    GLint ok = 0; glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) { char log[4096]; glGetProgramInfoLog(p, 4096, nullptr, log); fprintf(stderr, "Program link error:\n%s\n", log); }
    glDeleteShader(v); glDeleteShader(f);
    return p;
}
static GLuint makeTex(int w, int h, const uint8_t* data, bool repeatU) {
    GLuint t; glGenTextures(1, &t); glBindTexture(GL_TEXTURE_2D, t);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
    glGenerateMipmap(GL_TEXTURE_2D);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, repeatU ? GL_REPEAT : GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    return t;
}

// =====================================================================================
//  App state + input
// =====================================================================================
struct App {
    float yaw = 0.6f, pitch = 0.22f, dist = 30.f;
    bool down = false, dragged = false; double px = 0, py = 0, lx = 0, ly = 0;
    bool click = false; double cx = 0, cy = 0;
    int kind = 0; float vf = 0.55f; float dirSign = 1.f; int planeMode = 0;
    bool paused = false; float timeScale = 1.f;
    bool diskOn = true, beam = true, lensParticles = true, autoRes = true, autoOrbit = false;
    int scaleIdx = 3, densityIdx = 1;
    bool fsToggle = false, clearReq = false, resetReq = false;
} g;

static void printHelp() {
    printf("\n===== BLACK HOLE SIMULATOR - CONTROLS =====\n"
        " Left mouse drag ........ rotate camera around the black hole\n"
        " Mouse wheel / W,S ...... zoom in / out\n"
        " Arrow keys ............. rotate camera\n"
        " Left CLICK (no drag) ... place a body at that point\n"
        " 1 / 2 / 3 .............. body type: STAR / GIANT (big, easy to shred) / PLANET\n"
        " Z / X .................. initial speed (0 = straight plunge, 1 = circular orbit)\n"
        " T ...................... flip orbit direction\n"
        " M ...................... placement plane: equatorial (disk plane) / screen plane\n"
        " SPACE .................. pause     , / . : slower / faster time\n"
        " C ...................... clear bodies     R : reset everything (also black hole mass)\n"
        " B ...................... Doppler beaming     D : accretion disk on/off     L : particle lensing on/off\n"
        " O ...................... auto-orbit camera\n"
        " A ...................... auto resolution on/off     [ ] : manual render scale\n"
        " N ...................... particle density for NEW bodies (low/normal/high)\n"
        " F ...................... fullscreen     H : this help     ESC : quit\n"
        "============================================\n\n");
}
static void onMouseButton(GLFWwindow* w, int button, int action, int) {
    if (button != GLFW_MOUSE_BUTTON_LEFT) return;
    double mx, my; glfwGetCursorPos(w, &mx, &my);
    if (action == GLFW_PRESS) { g.down = true; g.dragged = false; g.px = g.lx = mx; g.py = g.ly = my; }
    else if (action == GLFW_RELEASE) { if (g.down && !g.dragged) { g.click = true; g.cx = mx; g.cy = my; } g.down = false; }
}
static void onCursor(GLFWwindow*, double x, double y) {
    if (!g.down) return;
    double dx = x - g.lx, dy = y - g.ly; g.lx = x; g.ly = y;
    if (!g.dragged && (fabs(x - g.px) + fabs(y - g.py)) > 5.0) g.dragged = true;
    if (g.dragged) { g.yaw -= (float)dx * 0.006f; g.pitch = clampf(g.pitch + (float)dy * 0.006f, -1.45f, 1.45f); }
}
static void onScroll(GLFWwindow*, double, double yoff) {
    g.dist *= (yoff > 0) ? 0.9f : 1.1f; g.dist = clampf(g.dist, 4.f, 220.f);
}
static void onKey(GLFWwindow* w, int key, int, int action, int) {
    if (action != GLFW_PRESS) return;
    switch (key) {
    case GLFW_KEY_ESCAPE: glfwSetWindowShouldClose(w, 1); break;
    case GLFW_KEY_1: g.kind = 0; break;
    case GLFW_KEY_2: g.kind = 1; break;
    case GLFW_KEY_3: g.kind = 2; break;
    case GLFW_KEY_Z: g.vf = std::max(0.f, g.vf - 0.1f); break;
    case GLFW_KEY_X: g.vf = std::min(1.2f, g.vf + 0.1f); break;
    case GLFW_KEY_T: g.dirSign = -g.dirSign; break;
    case GLFW_KEY_M: g.planeMode ^= 1; break;
    case GLFW_KEY_C: g.clearReq = true; break;
    case GLFW_KEY_R: g.resetReq = true; break;
    case GLFW_KEY_SPACE: g.paused = !g.paused; break;
    case GLFW_KEY_COMMA: g.timeScale = std::max(0.125f, g.timeScale * 0.5f); break;
    case GLFW_KEY_PERIOD: g.timeScale = std::min(8.f, g.timeScale * 2.f); break;
    case GLFW_KEY_B: g.beam = !g.beam; break;
    case GLFW_KEY_D: g.diskOn = !g.diskOn; break;
    case GLFW_KEY_L: g.lensParticles = !g.lensParticles; break;
    case GLFW_KEY_O: g.autoOrbit = !g.autoOrbit; break;
    case GLFW_KEY_A: g.autoRes = !g.autoRes; break;
    case GLFW_KEY_LEFT_BRACKET: g.autoRes = false; g.scaleIdx = std::max(0, g.scaleIdx - 1); break;
    case GLFW_KEY_RIGHT_BRACKET: g.autoRes = false; g.scaleIdx = std::min(8, g.scaleIdx + 1); break;
    case GLFW_KEY_N: g.densityIdx = (g.densityIdx + 1) % 3; break;
    case GLFW_KEY_F: g.fsToggle = true; break;
    case GLFW_KEY_H: printHelp(); break;
    default: break;
    }
}

// =====================================================================================
//  main
// =====================================================================================
int main() {
    if (!glfwInit()) { fprintf(stderr, "GLFW init failed\n"); return 1; }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif
    GLFWwindow* win = glfwCreateWindow(1280, 720, "Black Hole Simulator", nullptr, nullptr);
    if (!win) { fprintf(stderr, "Cannot create an OpenGL 3.3 window (update your graphics driver)\n"); glfwTerminate(); return 1; }
    glfwMakeContextCurrent(win);
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) { fprintf(stderr, "GLAD init failed\n"); return 1; }
    glfwSwapInterval(1);
    glfwSetMouseButtonCallback(win, onMouseButton);
    glfwSetCursorPosCallback(win, onCursor);
    glfwSetScrollCallback(win, onScroll);
    glfwSetKeyCallback(win, onKey);

    int hw = (int)std::thread::hardware_concurrency(); if (hw <= 0) hw = 4;
    ThreadPool pool(std::max(0, std::min(hw, 4) - 1));

    // ---- textures: real images if present, otherwise procedural ones -----------------------
    Image sky; bool skyLoaded = false;
    const char* skyNames[] = { "space.jpg", "space.png", "space.bmp", "space.tga", "space.ppm" };
    for (const char* nme : skyNames) if (fileExists(nme) && loadImageFile(nme, sky) && sky.ok()) { skyLoaded = true; printf("Sky image: %s (%dx%d)\n", nme, sky.w, sky.h); break; }
    if (skyLoaded) { while (sky.w > 2048) halveImage(sky); flipRows(sky); }
    else { printf("No space.* image found -> procedural sky. (Put space.jpg/png/bmp/tga/ppm next to the exe)\n"); sky.w = 1024; sky.h = 512; genSky(sky.px, sky.w, sky.h, pool); }

    Image disk; bool diskLoaded = false;
    const char* diskNames[] = { "disk.png", "disk.jpg", "disk.bmp", "disk.tga", "disk.ppm" };
    for (const char* nme : diskNames) if (fileExists(nme) && loadImageFile(nme, disk) && disk.ok()) { diskLoaded = true; printf("Disk image: %s (%dx%d)\n", nme, disk.w, disk.h); break; }
    if (diskLoaded) {
        while (disk.w > 2048) halveImage(disk);
        if (!disk.alpha) for (size_t i = 0; i < disk.px.size(); i += 4) { int m = std::max(disk.px[i], std::max(disk.px[i + 1], disk.px[i + 2])); disk.px[i + 3] = (uint8_t)std::min(255, (int)(m * 1.6f)); }
        flipRows(disk);
    }
    else { printf("No disk.* image found -> procedural disk texture.\n"); disk.w = 1024; disk.h = 256; genDisk(disk.px, disk.w, disk.h, pool); }

    GLuint texSky = makeTex(sky.w, sky.h, sky.px.data(), true);
    GLuint texDisk = makeTex(disk.w, disk.h, disk.px.data(), true);

    // ---- programs --------------------------------------------------------------------------
    GLuint progLens = makeProgram(VS_FULL, FS_LENS);
    GLuint progBlit = makeProgram(VS_FULL, FS_BLIT);
    GLuint progPart = makeProgram(VS_PART, FS_PART);
    GLuint vaoEmpty; glGenVertexArrays(1, &vaoEmpty);

        GLuint vaoS, vboS, vaoG, vboG;
    glGenVertexArrays(1, &vaoS); glGenBuffers(1, &vboS);
    glGenVertexArrays(1, &vaoG); glGenBuffers(1, &vboG);
    auto setupPartVAO = [](GLuint vao, GLuint vbo) {
        glBindVertexArray(vao); glBindBuffer(GL_ARRAY_BUFFER, vbo);
        const int NC[6] = {3, 3, 3, 1, 1, 1}, OFF[6] = {0, 3, 6, 9, 10, 11};
        for (int a = 0; a < 6; a++) {
            glEnableVertexAttribArray(a);
            glVertexAttribPointer(a, NC[a], GL_FLOAT, GL_FALSE, sizeof(Vtx), (void*)(size_t)(OFF[a] * sizeof(float)));
            glVertexAttribDivisor(a, 2);          // two instances (lens images) per particle
        }
        glBindVertexArray(0);
    };
    setupPartVAO(vaoS, vboS); setupPartVAO(vaoG, vboG);

    // uniform locations
        // uniform locations
    auto L = [&](GLuint p, const char* n) { return glGetUniformLocation(p, n); };
    glUseProgram(progLens);
    glUniform1i(L(progLens, "uSky"), 0); glUniform1i(L(progLens, "uDisk"), 1);
    GLint uCam = L(progLens, "uCam"), uRight = L(progLens, "uRight"), uUp = L(progLens, "uUp"), uFwd = L(progLens, "uFwd");
    GLint uTanH = L(progLens, "uTanH"), uAspect = L(progLens, "uAspect"), uRsL = L(progLens, "uRs"), uTime = L(progLens, "uTime");
    GLint uFlare = L(progLens, "uFlare"), uSkyLod = L(progLens, "uSkyLod"), uDiskGain = L(progLens, "uDiskGain");
    GLint uFeedL = L(progLens, "uFeed"), uMassNL = L(progLens, "uMassN"), uJetL = L(progLens, "uJet");
    GLint uProcStars = L(progLens, "uProcStars"), uDiskOn = L(progLens, "uDiskOn"), uBeam = L(progLens, "uBeam");
    glUseProgram(progBlit); glUniform1i(L(progBlit, "uTex"), 0);
    GLint pView = L(progPart, "uView"), pProj = L(progPart, "uProj"), pCam = L(progPart, "uCam"), pRs = L(progPart, "uRs");
    GLint pPx = L(progPart, "uPx"), pLens = L(progPart, "uLens"), pMode = L(progPart, "uMode"), pTrail = L(progPart, "uTrail");
    // render target (reduced resolution) + GPU timer queries for adaptive quality
    GLuint fbo = 0, fboTex = 0; int fw = 0, fh = 0;
    GLuint queries[2]; glGenQueries(2, queries); bool qPending[2] = { false, false }; int qi = 0; float gpuMs = 0.f; bool gpuValid = false;
    const float SCALES[9] = { 0.25f, 0.33f, 0.42f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f };
    int adaptCounter = 0;

    glEnable(GL_PROGRAM_POINT_SIZE);
    glClearColor(0, 0, 0, 1);

    Sim sim;
    std::vector<Vtx> solidV, glowV;
    printHelp();

    double last = glfwGetTime(), titleT = 0; int frames = 0; float fpsShown = 0.f;
    int winX = 100, winY = 100, winW = 1280, winH = 720;

    while (!glfwWindowShouldClose(win)) {
        glfwPollEvents();
        double now = glfwGetTime(); float dt = (float)(now - last); last = now; dt = std::min(dt, 0.1f);

        int fbw, fbh; glfwGetFramebufferSize(win, &fbw, &fbh);
        if (fbw <= 0 || fbh <= 0) { glfwWaitEventsTimeout(0.1); continue; }
        int W, H; glfwGetWindowSize(win, &W, &H); if (W <= 0) W = 1; if (H <= 0) H = 1;

        // fullscreen toggle
        if (g.fsToggle) {
            g.fsToggle = false;
            if (!glfwGetWindowMonitor(win)) {
                glfwGetWindowPos(win, &winX, &winY); glfwGetWindowSize(win, &winW, &winH);
                GLFWmonitor* mon = glfwGetPrimaryMonitor(); const GLFWvidmode* vm = glfwGetVideoMode(mon);
                glfwSetWindowMonitor(win, mon, 0, 0, vm->width, vm->height, vm->refreshRate);
            }
            else glfwSetWindowMonitor(win, nullptr, winX, winY, winW, winH, 0);
            glfwSwapInterval(1);
            continue;
        }

        // continuous camera keys
        float rotSpd = 1.4f * dt;
        if (glfwGetKey(win, GLFW_KEY_LEFT) == GLFW_PRESS) g.yaw -= rotSpd;
        if (glfwGetKey(win, GLFW_KEY_RIGHT) == GLFW_PRESS) g.yaw += rotSpd;
        if (glfwGetKey(win, GLFW_KEY_UP) == GLFW_PRESS) g.pitch = clampf(g.pitch + rotSpd, -1.45f, 1.45f);
        if (glfwGetKey(win, GLFW_KEY_DOWN) == GLFW_PRESS) g.pitch = clampf(g.pitch - rotSpd, -1.45f, 1.45f);
        if (glfwGetKey(win, GLFW_KEY_W) == GLFW_PRESS) g.dist = clampf(g.dist * (1.f - 1.2f * dt), 4.f, 220.f);
        if (glfwGetKey(win, GLFW_KEY_S) == GLFW_PRESS) g.dist = clampf(g.dist * (1.f + 1.2f * dt), 4.f, 220.f);
        if (g.autoOrbit) g.yaw += 0.12f * dt;

        // camera basis
        V3 eye{ g.dist * cosf(g.pitch) * sinf(g.yaw), g.dist * sinf(g.pitch), g.dist * cosf(g.pitch) * cosf(g.yaw) };
        V3 up0{ 0, 1, 0 };
        V3 fwd = norm3(eye * -1.f), right = norm3(cross3(fwd, up0)), up = cross3(right, fwd);
        const float fovy = 60.f * PI_F / 180.f, tanH = tanf(fovy * 0.5f);
        float aspect = (float)fbw / (float)fbh;

        // reset / clear
        if (g.clearReq) { sim.bodies.clear(); g.clearReq = false; }
        if (g.resetReq) { sim = Sim(); g.resetReq = false; }

        // click -> place a body
        if (g.click) {
            g.click = false;
            float nx = (float)(2.0 * g.cx / W - 1.0), ny = (float)(1.0 - 2.0 * g.cy / H);
            V3 d = norm3(fwd + right * (nx * tanH * aspect) + up * (ny * tanH));
            V3 P{ 0, 0, 0 }, N{ 0, 1, 0 }; bool ok = false;
            if (g.planeMode == 0 && fabsf(d.y) > 0.08f) {
                float t = -eye.y / d.y;
                if (t > 0.f && t < 800.f) { P = eye + d * t; N = { 0, 1, 0 }; ok = true; }
            }
            if (!ok) {
                float dn = dot3(d, fwd);
                if (dn > 1e-3f) { float t = g.dist / dn; P = eye + d * t; N = fwd; ok = true; }
            }
            float rs = sim.rs();
            if (ok && len3(P) > 1.3f * rs) {
                float r = len3(P); V3 rh = P * (1.f / r);
                V3 tang = cross3(N, rh);
                if (len3(tang) < 1e-3f) tang = cross3(right, rh);
                tang = norm3(tang) * g.dirSign;
                float GM = (float)sim.Mbh;
                float vc = sqrtf(GM * r) / std::max(r - rs, 0.2f);
                vc = std::min(vc, 0.8f * C_LIGHT);
                spawnBody(sim, g.kind, P, tang * (g.vf * vc), DENS[g.densityIdx]);
            }
        }

        // physics
        if (!g.paused) updatePhysics(sim, std::min(dt, 0.05f) * g.timeScale, pool);
        if (!g.paused) updateFeed(sim, std::min(dt, 0.05f) * g.timeScale);
        sim.flare *= expf(-dt * 0.8f);
        updateStats(sim);
        buildVerts(sim, solidV, glowV);

        // adaptive resolution (based on GPU time of the black hole pass)
        if (g.autoRes && gpuValid && ++adaptCounter >= 20) {
            adaptCounter = 0;
            if (gpuMs > 11.f && g.scaleIdx > 0) g.scaleIdx--;
            else if (g.scaleIdx < 8) {
                float ratio = SCALES[g.scaleIdx + 1] / SCALES[g.scaleIdx];
                if (gpuMs * ratio * ratio < 8.5f) g.scaleIdx++;
            }
        }
        float scale = SCALES[g.scaleIdx];
        int rw = std::max(64, (int)(fbw * scale)), rh = std::max(64, (int)(fbh * scale));
        if (rw != fw || rh != fh || !fbo) {
            if (!fbo) { glGenFramebuffers(1, &fbo); glGenTextures(1, &fboTex); }
            glBindTexture(GL_TEXTURE_2D, fboTex);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, rw, rh, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            glBindFramebuffer(GL_FRAMEBUFFER, fbo);
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, fboTex, 0);
            fw = rw; fh = rh;
        }

        // ---------------- pass 1: black hole (low-res) ----------------
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        glViewport(0, 0, rw, rh);
        glDisable(GL_DEPTH_TEST); glDisable(GL_BLEND);
        glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, texSky);
        glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, texDisk);
        glUseProgram(progLens);
        glUniform3f(uCam, eye.x, eye.y, eye.z);
        glUniform3f(uRight, right.x, right.y, right.z); glUniform3f(uUp, up.x, up.y, up.z); glUniform3f(uFwd, fwd.x, fwd.y, fwd.z);
        glUniform1f(uTanH, tanH); glUniform1f(uAspect, aspect); glUniform1f(uRsL, sim.rs());
        glUniform1f(uTime, (float)now); glUniform1f(uFlare, sim.flare);
        float lod = 0.f;
        { float tpp = sky.h * (fovy / PI_F) / (float)rh; lod = std::max(0.f, log2f(std::max(tpp, 1e-3f))); }
        glUniform1f(uSkyLod, lod); glUniform1f(uDiskGain, 2.4f);
        glUniform1f(uFeedL, clampf(sim.feed * 0.75f, 0.f, 1.7f));
        glUniform1f(uMassNL, clampf(sim.diskMass * 1.3f, 0.f, 1.2f));
        glUniform1f(uJetL, sim.jet);
        glUniform1i(uProcStars, skyLoaded ? 0 : 1); glUniform1i(uDiskOn, g.diskOn ? 1 : 0); glUniform1i(uBeam, g.beam ? 1 : 0);
        glBindVertexArray(vaoEmpty);
        glBeginQuery(GL_TIME_ELAPSED, queries[qi]);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        glEndQuery(GL_TIME_ELAPSED);
        qPending[qi] = true;
        {
            int prev = qi ^ 1;
            if (qPending[prev]) {
                GLint av = 0; glGetQueryObjectiv(queries[prev], GL_QUERY_RESULT_AVAILABLE, &av);
                if (av) {
                    GLuint ns = 0; glGetQueryObjectuiv(queries[prev], GL_QUERY_RESULT, &ns);
                    float ms = ns / 1.0e6f; gpuMs = gpuValid ? gpuMs * 0.9f + 0.1f * ms : ms; gpuValid = true; qPending[prev] = false;
                }
            }
            qi ^= 1;
        }

        // ---------------- pass 2: upscale to screen ----------------
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glViewport(0, 0, fbw, fbh);
        glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, fboTex);
        glUseProgram(progBlit);
        glDrawArrays(GL_TRIANGLES, 0, 3);

        // ---------------- pass 3: bodies (particles) ----------------
        if (!solidV.empty() || !glowV.empty()) {
            float view[16], proj[16];
            mat4LookAt(view, eye, V3{ 0, 0, 0 }, up0); mat4Perspective(proj, fovy, aspect, 0.1f, 3000.f);
            glUseProgram(progPart);
            glUniformMatrix4fv(pView, 1, GL_FALSE, view); glUniformMatrix4fv(pProj, 1, GL_FALSE, proj);
            glUniform3f(pCam, eye.x, eye.y, eye.z); glUniform1f(pRs, sim.rs());
            glUniform1f(pPx, (float)fbh / (2.f * tanH)); glUniform1i(pLens, g.lensParticles ? 1 : 0);
            glUniform1f(pTrail, 0.030f);
            glDepthMask(GL_TRUE); glClear(GL_DEPTH_BUFFER_BIT); glEnable(GL_DEPTH_TEST); glDepthFunc(GL_LESS);
            if (!solidV.empty()) {
                glBindVertexArray(vaoS); glBindBuffer(GL_ARRAY_BUFFER, vboS);
                glBufferData(GL_ARRAY_BUFFER, solidV.size() * sizeof(Vtx), solidV.data(), GL_STREAM_DRAW);
                glDisable(GL_BLEND); glUniform1i(pMode, 1);
                glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, (GLsizei)(solidV.size() * 2));
            }
            if (!glowV.empty()) {
                glBindVertexArray(vaoG); glBindBuffer(GL_ARRAY_BUFFER, vboG);
                glBufferData(GL_ARRAY_BUFFER, glowV.size() * sizeof(Vtx), glowV.data(), GL_STREAM_DRAW);
                glDepthMask(GL_FALSE); glEnable(GL_BLEND); glBlendFunc(GL_ONE, GL_ONE); glUniform1i(pMode, 0);
                glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, (GLsizei)(glowV.size() * 2));
            }
            glDepthMask(GL_TRUE); glDisable(GL_BLEND); glDisable(GL_DEPTH_TEST);
            glBindVertexArray(0);
        }

        glfwSwapBuffers(win);

        // title bar HUD
        frames++; titleT += dt;
        if (titleT >= 0.5) {
            fpsShown = (float)(frames / titleT); frames = 0; titleT = 0;
            int parts = 0; for (auto& b : sim.bodies) parts += b.count();
            char buf[320];
            snprintf(buf, sizeof(buf), "BH | %.0f FPS | res %d%%%s | GPU %.1fms | bodies %d (%d p) | %s v=%.1f%s | x%.2f%s | M=%.1f rs=%.2f | feed %.2f jet %.2f",
                fpsShown, (int)(scale * 100), g.autoRes ? " auto" : "", gpuMs, (int)sim.bodies.size(), parts,
                KINDS[g.kind].name, g.vf, g.dirSign > 0 ? "" : " (rev)", g.timeScale, g.paused ? " PAUSED" : "",
                sim.Mbh, sim.rs(), sim.feed, sim.jet);
            glfwSetWindowTitle(win, buf);
        }
    }

    glfwDestroyWindow(win);
    glfwTerminate();
    return 0;
}
