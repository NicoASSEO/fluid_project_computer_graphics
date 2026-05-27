#define _CRT_SECURE_NO_WARNINGS 1

#include <iostream>
#include <sstream>
#include <vector>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <sys/stat.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <lbfgs.h>

double sqr(double x) { return x * x; };

class Vector {
public:
    explicit Vector(double x = 0, double y = 0) {
        data[0] = x;
        data[1] = y;
    }
    double norm2() const {
        return data[0] * data[0] + data[1] * data[1];
    }
    double norm() const {
        return sqrt(norm2());
    }
    void normalize() {
        double n = norm();
        data[0] /= n;
        data[1] /= n;
    }
    double operator[](int i) const { return data[i]; };
    double& operator[](int i) { return data[i]; };
    double data[2];
};

Vector operator+(const Vector& a, const Vector& b) {
    return Vector(a[0] + b[0], a[1] + b[1]);
}
Vector operator-(const Vector& a, const Vector& b) {
    return Vector(a[0] - b[0], a[1] - b[1]);
}
Vector operator*(const double a, const Vector& b) {
    return Vector(a * b[0], a * b[1]);
}
Vector operator*(const Vector& a, const double b) {
    return Vector(a[0] * b, a[1] * b);
}
Vector operator/(const Vector& a, const double b) {
    return Vector(a[0] / b, a[1] / b);
}
double dot(const Vector& a, const Vector& b) {
    return a[0] * b[0] + a[1] * b[1];
}
double cross(const Vector& a, const Vector& b) {
    return a[0] * b[1] - a[1] * b[0];
}


class Polygon {
public:

    double area() {
        if (vertices.size() < 3) return 0;
        double sum = 0;
        int m = (int)vertices.size();
        for (int i = 0; i < m; i++) {
            int j = (i + 1) % m;
            sum += vertices[i][0] * vertices[j][1] - vertices[j][0] * vertices[i][1];
        }
        return 0.5 * fabs(sum);
    }

    Vector centroid() {
        if (vertices.size() < 3) return Vector(0, 0);

        // TODO Lab 3
        // Compute the centroid of the polygon
        double A = 0;
        double cx = 0;
        double cy = 0;
        int m = (int)vertices.size();
        for (int i = 0; i < m; i++) {
            int j = (i + 1) % m;
            double cross_ij = vertices[i][0] * vertices[j][1] - vertices[j][0] * vertices[i][1];
            A += cross_ij;
            cx += (vertices[i][0] + vertices[j][0]) * cross_ij;
            cy += (vertices[i][1] + vertices[j][1]) * cross_ij;
        }
        A *= 0.5;
        if (fabs(A) < 1e-14) return Vector(0, 0);
        return Vector(cx / (6.0 * A), cy / (6.0 * A));
    }

    double integral_square_distance(const Vector& Pi) {
        double total = 0;
        int m = vertices.size();
        for (int i = 0; i < m; i++) {
            Vector B = vertices[i];
            Vector C = vertices[(i + 1) % m];
            double areaT = 0.5 * fabs(cross(B - Pi, C - Pi));
            Vector b = B - Pi;
            Vector c = C - Pi;
            total += areaT / 6.0 * (dot(b, b) + dot(c, c) + dot(b, c));
        }
        return total;
    }

    std::vector<Vector> vertices;
};


void save_frame(const std::vector<Polygon>& cells, std::string filename, int frameid = 0) {
    constexpr int W = 800, H = 800;
    constexpr double edge_width = 2.0;
    constexpr double edge_width2 = edge_width * edge_width;
    

    std::vector<unsigned char> inside(W * H, 0), edge(W * H, 0);

#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < (int)cells.size(); ++i) {
        const auto& V = cells[i].vertices;
        const int n = (int)V.size();
        if (n < 3) continue;

        std::vector<double> xs(n), ys(n);
        double xmin = 1e30, ymin = 1e30, xmax = -1e30, ymax = -1e30;
        for (int j = 0; j < n; ++j) {
            xs[j] = V[j][0] * W;
            ys[j] = V[j][1] * H;
            xmin = std::min(xmin, xs[j]);
            ymin = std::min(ymin, ys[j]);
            xmax = std::max(xmax, xs[j]);
            ymax = std::max(ymax, ys[j]);
        }

        int x0 = std::max(0, (int)std::floor(xmin - edge_width));
        int y0 = std::max(0, (int)std::floor(ymin - edge_width));
        int x1 = std::min(W - 1, (int)std::ceil(xmax + edge_width));
        int y1 = std::min(H - 1, (int)std::ceil(ymax + edge_width));
        for (int y = y0; y <= y1; ++y) {
            for (int x = x0; x <= x1; ++x) {
                const double px = x + 0.5, py = y + 0.5;

                int prev_sign = 0;
                bool isInside = true;
                bool isEdge = false;

                for (int j = 0; j < n; ++j) {
                    int k = (j + 1) % n;

                    double ax = xs[j], ay = ys[j];
                    double bx = xs[k], by = ys[k];
                    double dx = bx - ax, dy = by - ay;
                    double qx = px - ax, qy = py - ay;

                    double det = qx * dy - qy * dx;
                    int s = (det > 1e-12) - (det < -1e-12);

                    if (s != 0) {
                        if (prev_sign != 0 && s != prev_sign) {
                            isInside = false;
                            break;
                        }
                        prev_sign = s;
                    }

                    double len2 = dx * dx + dy * dy;
                    double dprod = qx * dx + qy * dy;
                    if (dprod >= 0.0 && dprod <= len2 && det * det <= edge_width2 * len2)
                        isEdge = true;
                }

                if (isInside) {
                    int id = (H - 1 - y) * W + x;
                    inside[id] = 1;
                    if (isEdge) edge[id] = 1;
                }
            }
        }
    }

    std::vector<unsigned char> image(W * H * 3, 255);

#pragma omp parallel for
    for (int i = 0; i < W * H; ++i) {
        if (edge[i]) {
            image[3 * i + 0] = 0;
            image[3 * i + 1] = 0;
            image[3 * i + 2] = 0;
        }
        else if (inside[i]) {
            image[3 * i + 0] = 0;
            image[3 * i + 1] = 0;
            image[3 * i + 2] = 255;
        }
    }

    std::ostringstream os;
    os << filename << frameid << ".png";
    stbi_write_png(os.str().c_str(), W, H, 3, image.data(), W * 3);
}


class VoronoiDiagram {

public:

    VoronoiDiagram() {
    };

    void compute() {
        cells.resize(points.size());
        if (weights.size() != points.size()) {
            weights.resize(points.size(), 0.0);
        }

#pragma omp parallel for
        for (int i = 0; i < (int)points.size(); i++) {
            Polygon cell;
            cell.vertices.push_back(Vector(0, 0));
            cell.vertices.push_back(Vector(1, 0));
            cell.vertices.push_back(Vector(1, 1));
            cell.vertices.push_back(Vector(0, 1));

            for (int j = 0; j < (int)points.size(); j++) {
                if (i == j) {
                    continue;
                }
                cell = clip_by_bisector(cell, points[i], points[j], weights[i], weights[j]);
                if (cell.vertices.empty()) {
                    break;
                }
            }

            double dr2 = weights[i] - w_air;
            if (!cell.vertices.empty() && dr2 > 1e-12) {
                cell = clip_by_disk(cell, points[i], sqrt(dr2));
            } else if (!cell.vertices.empty()) {
                cell.vertices.clear();
            }

            cells[i] = cell;
        }
    }


    static Polygon clip_by_edge(const Polygon& V, const Vector& u, const Vector& v) {
        // TODO Lab 3 (fluids)
        // Clip a polygon by an edge defined by vertices u and v
        // Will be used to clip a polygon (a cell) by all the edges of a (discretized) disk
        Polygon result;
        result.vertices.reserve(V.vertices.size() + 1);
        if (V.vertices.empty()) {
            return result;
        }

        Vector edge = v - u;
        Vector N(-edge[1], edge[0]);

        for (int i = 0; i < (int)V.vertices.size(); i++) {
            Vector A = V.vertices[i];
            Vector B = V.vertices[(i + 1) % V.vertices.size()];

            bool A_inside = dot(A - u, N) >= 0;
            bool B_inside = dot(B - u, N) >= 0;

            if (A_inside != B_inside) {
                double denom = dot(B - A, N);
                double t = dot(u - A, N) / denom;
                result.vertices.push_back(A + t * (B - A));
            }

            if (B_inside) {
                result.vertices.push_back(B);
            }
        }

        return result;
    }

    static Polygon clip_by_disk(const Polygon& V, const Vector& center, double radius) {
        Polygon cell = V;
        const int sides = 64;
        for (int k = 0; k < sides; k++) {
            double a0 = 2.0 * M_PI * k / sides;
            double a1 = 2.0 * M_PI * (k + 1) / sides;
            Vector u(center[0] + radius * cos(a0), center[1] + radius * sin(a0));
            Vector v(center[0] + radius * cos(a1), center[1] + radius * sin(a1));
            cell = clip_by_edge(cell, u, v);
            if (cell.vertices.empty()) {
                break;
            }
        }
        return cell;
    }

    static Polygon clip_by_bisector(const Polygon& V, const Vector& Pi, const Vector& Pj, double w0, double wi) {
        Polygon result;
        result.vertices.reserve(V.vertices.size() + 1);
        if (V.vertices.empty()) {
            return result;
        }

        Vector d = Pj - Pi;
        double d2 = d.norm2();
        if (d2 < 1e-14) {
            return V;
        }

        Vector M = (Pi + Pj) / 2;
        Vector Mp = M + (w0 - wi) / (2 * d2) * d;

        for (int i = 0; i < (int)V.vertices.size(); i++) {
            Vector A = V.vertices[i];
            Vector B = V.vertices[(i + 1) % V.vertices.size()];

            bool A_inside = dot(A - Mp, d) <= 0;
            bool B_inside = dot(B - Mp, d) <= 0;

            if (A_inside != B_inside) {
                double denom = dot(B - A, Pi - Pj);
                double t = dot(Mp - A, Pi - Pj) / denom;
                result.vertices.push_back(A + t * (B - A));
            }

            if (B_inside) {
                result.vertices.push_back(B);
            }
        }

        return result;
    }


    std::vector<Vector> points;
    std::vector<double> weights;
    std::vector<Polygon> cells;
    double w_air = 0.0;

};


class OptimalTransport {

public:
    OptimalTransport() {};

    void optimize();

    VoronoiDiagram vor;
    double fluid_volume = 0.0;
    double w_air = 0.0;
};


static lbfgsfloatval_t evaluate(
    void* instance,
    const lbfgsfloatval_t* x,
    lbfgsfloatval_t* g,
    const int n,
    const lbfgsfloatval_t step
)
{
    OptimalTransport* ot = (OptimalTransport*)(instance);
    int N = (int)ot->vor.points.size();

    memcpy(&ot->vor.weights[0], x, N * sizeof(x[0]));
    ot->w_air = x[N];
    ot->vor.w_air = ot->w_air;
    ot->vor.compute();

    // Lab 2 (Optimal transport) : compute the function to be minimized (fx) and its gradient (g[i], i=0..n-1)
    // Lab 3 (fluid) : adapt these functions to support partial optimal transport (now "n" has been increased by 1 to account for the air variable)

    lbfgsfloatval_t fx = 0.0;
    double lambda = ot->fluid_volume / N;
    double desired_air = 1.0 - ot->fluid_volume;
    double fluid_area = 0.0;

    for (int i = 0; i < N; i++) {
        double A = ot->vor.cells[i].area();
        double I = ot->vor.cells[i].integral_square_distance(ot->vor.points[i]);
        double wi = ot->vor.weights[i];

        fx += I - wi * A + lambda * wi;
        g[i] = A - lambda;
        fluid_area += A;
    }

    double estimated_air = 1.0 - fluid_area;
    fx += x[N] * (desired_air - estimated_air);
    g[N] = estimated_air - desired_air;
    fx = -fx;

    return fx;
}


void OptimalTransport::optimize() {
    int N = (int)vor.points.size();
    std::vector<double> x(N + 1);
    for (int i = 0; i < N; i++) {
        x[i] = vor.weights[i];
    }
    x[N] = w_air;

    lbfgsfloatval_t fx;
    lbfgs_parameter_t param;
    lbfgs_parameter_init(&param);
    param.max_iterations = 100;

    lbfgs(N + 1, &x[0], &fx, evaluate, NULL, (void*)this, &param);

    vor.weights.resize(N);
    for (int i = 0; i < N; i++) {
        vor.weights[i] = x[i];
    }
    w_air = x[N];
    vor.w_air = w_air;
    vor.compute();
}


class Fluid {
public:
    Fluid(int N_particles = 80) : N_particles(N_particles) {
        double cx = 0.5;
        double cy = 0.72;
        double r = 0.155;
        fluid_volume = 0.20;

        particles.resize(N_particles);
        velocities.resize(N_particles, Vector(0, 0));

        srand(0);
        for (int i = 0; i < N_particles; i++) {
            double u = rand() / (double)RAND_MAX;
            double v = rand() / (double)RAND_MAX;
            double rr = r * sqrt(u);
            double theta = 2.0 * M_PI * v;
            particles[i] = Vector(cx + rr * cos(theta), cy + rr * sin(theta));
        }

        ot.fluid_volume = fluid_volume;
        ot.w_air = 0.0;
        ot.vor.w_air = 0.0;
        ot.vor.weights.assign(N_particles, 0.0);
        ot.vor.points = particles;
        ot.optimize();
    }

    void time_step(double dt) {
        double epsilon2 = 0.004 * 0.004;
        Vector grav(0, -40.0);
        double m_i = 200.0;

        ot.vor.points = particles;
        ot.optimize();

        for (int i = 0; i < N_particles; i++) {
            Vector Fspring(0, 0);
            if (ot.vor.cells[i].area() > 1e-10) {
                Fspring = (ot.vor.cells[i].centroid() - particles[i]) / epsilon2;
            }
            Vector F = Fspring + m_i * grav;
            velocities[i] = velocities[i] + (dt / m_i) * F;
            particles[i] = particles[i] + dt * velocities[i];

            if (particles[i][0] < 0) {
                particles[i][0] = -particles[i][0];
                velocities[i][0] = -velocities[i][0];
            }
            if (particles[i][0] > 1) {
                particles[i][0] = 2 - particles[i][0];
                velocities[i][0] = -velocities[i][0];
            }
            if (particles[i][1] < 0) {
                particles[i][1] = -particles[i][1];
                velocities[i][1] = -velocities[i][1];
            }
            if (particles[i][1] > 1) {
                particles[i][1] = 2 - particles[i][1];
                velocities[i][1] = -velocities[i][1];
            }
        }
    }

    void run_simulation() {
        mkdir("frames", 0755);
        double dt = 0.002;
        for (int i = 0; i < 1000; i++) {
            time_step(dt);
            save_frame(ot.vor.cells, "frames/frame", i);
        }
    }

    int N_particles;

    OptimalTransport ot;
    std::vector<Vector> particles;
    std::vector<Vector> velocities;
    double fluid_volume;
};


int main() {
    Fluid fluid(500);
    fluid.run_simulation();
    return 0;
}
