// rdm_similarity.cpp
//
// This program computes whitened unbiased cosine similarity of distance matrices. 
// It takes distance matrices, and distance covariance matrices as input, pools the 
// covariances of pairs of distance matrices and whitens them before computing cosine
// similarity. This script is designed to work on the output of the custom RDM nipype 
// class.
//
// drated by chatGPT-4o. Completed by Bogdan Petre, 2025

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <stdexcept>

#include <Eigen/Dense>
#include <nlohmann/json.hpp>
//#include <cblas.h>

using namespace std;
using namespace Eigen;
using json = nlohmann::json;

template <typename Real_T>
inline Real_T regularization_epsilon();

template <>
inline float regularization_epsilon<float>() { return 1e-6f; }

template <>
inline double regularization_epsilon<double>() { return 1e-12; }

using MatrixT = Eigen::Matrix<REAL_T, Dynamic, Dynamic>;
using VectorT = Eigen::Matrix<REAL_T, Dynamic, 1>;

struct MetaData {
    std::string format;
    int shape_per_matrix_0;
    size_t n_regions;
    REAL_T dof;
};

MetaData parse_metadata(const string& filename) {
    ifstream file(filename);
    if (!file) throw runtime_error("Failed to open metadata file: " + filename);

    json j;
    file >> j;

    MetaData meta;
    meta.shape_per_matrix_0 = j.at("shape").at(0);
    meta.n_regions = j.at("n_regions");
    meta.dof = j.at("dof");
    meta.format = j.at("format");
    return meta;
}

MatrixT unpack_lower_triangle(const vector<REAL_T>& data, int dim) {
    MatrixT mat(dim, dim);
    mat.setZero();
    int index = 0;
    for (int i = 0; i < dim; ++i) {
        for (int j = i; j < dim; ++j) {
            mat(j, i) = data[index];
            if (i != j) mat(i, j) = data[index];
            ++index;
        }
    }
    return mat;
}

vector<REAL_T> read_region_cov_matrix(const string& filename, size_t region_index, size_t REAL_Ts_per_matrix) {
    ifstream fin(filename, ios::binary);
    if (!fin) throw runtime_error("Failed to open binary file " + filename);
    size_t offset = region_index * REAL_Ts_per_matrix * sizeof(REAL_T);
    fin.seekg(offset, ios::beg);
    vector<REAL_T> buffer(REAL_Ts_per_matrix);
    fin.read(reinterpret_cast<char*>(buffer.data()), REAL_Ts_per_matrix * sizeof(REAL_T));
    return buffer;
}

MatrixT load_csv_column(const string& filename, size_t col, size_t rows) {
    ifstream fin(filename);
    if (!fin) throw runtime_error("Failed to open " + filename);
    MatrixT mat(rows, 1);
    string line;
    size_t i = 0;
    while (getline(fin, line) && i < rows) {
        stringstream ss(line);
        string cell;
        for (size_t j = 0; j <= col; ++j) {
            if (!getline(ss, cell, ','))
                throw runtime_error("Malformed CSV row in " + filename);
        }
        try {
            mat(i, 0) = stof(cell);
        } catch (const std::exception& e) {
            cerr << "Warning: failed to parse value from column " << col
                 << " in line " << i << " of " << filename
                 << ": '" << cell << "' (" << e.what() << ")\n";
            mat(i, 0) = NAN; // or some sentinel value
        }
        ++i;
    }
    return mat;
}

MatrixT pooled_covariance(const MatrixT& A, REAL_T dofA, const MatrixT& B, REAL_T dofB) {
    return (A * dofA + B * dofB) / (dofA + dofB);
}

extern "C" {
    void spotrf_(char* uplo, int* n, float* a, int* lda, int* info);
    void dpotrf_(char* uplo, int* n, double* a, int* lda, int* info);
}


/* BEGIN: Task Balancing Code */

// expand lower diagonal vectorization of a matrix into a symmetric matrix
MatrixT vector_to_matrix(const VectorT& vec, int dim) {
    MatrixT mat(dim, dim);
    mat.setZero();
    int index = 0;
    for (int i = 0; i < dim; ++i) {
        for (int j = i + 1; j < dim; ++j) {
            mat(i, j) = mat(j, i) = vec(index++);
        }
    }
    return mat;
}

// vectorize symmetric matrix using lower triangular part
VectorT matrix_to_vector(const MatrixT& mat) {
    int dim = mat.rows();
    VectorT vec(dim * (dim - 1) / 2);
    int index = 0;
    for (int i = 0; i < dim; ++i) {
        for (int j = i + 1; j < dim; ++j) {
            vec(index++) = mat(i, j);
        }
    }
    return vec;
}

// A replicationPlan can be precomputed before iterating over regions and reused.
// These are operations related to expanding the RDMs for the sake of balancing
// cosine similarity computaitons and are identical for all regions.
struct ReplicationPlan {
    // replicate_ids contains indices of conditions, each repeated as many times
    // as that condition needs to be replicated to balance across tasks
    vector<int> replicate_ids;
    // block_start_stop delimintes tasks in replicate_ids
    vector<pair<int, int>> block_start_stop;
    vector<int> block_labels;
};

ReplicationPlan make_replication_plan(const vector<int>& task_ids) {
    ReplicationPlan plan;
    unordered_map<int, int> count_map;
    for (int id : task_ids) count_map[id]++;

    // find least common multiple (lcm) of all count_map elements using Euclidean 
    // algorithm. This becomes the total number of replicates we need of each task.
    int lcm = 1;
    for (const auto& [_, count] : count_map) {
        int a = lcm, b = count;
        while (b != 0) {
            int t = b;
            b = a % b;
            a = t;
        }
        lcm = lcm * count / a;
    }

    // By dividing the lcm by the number of conditions/task we get the number of 
    // replicates we need of each condition to balance conditions across tasks.
    unordered_map<int, int> replicates_per_class;
    for (const auto& [id, count] : count_map)
        replicates_per_class[id] = lcm / count;

    vector<int> ids;
    for (size_t i = 0; i < task_ids.size(); ++i) {
        for (int r = 0; r < replicates_per_class[task_ids[i]]; ++r) {
            ids.push_back(i);
        }
    }
    plan.replicate_ids = ids;

    unordered_map<int, pair<int, int>> block_map;
    int pos = 0;
    for (const auto& [label, count] : replicates_per_class) {
        int start = pos;
        int end = start + count_map[label] * count - 1;
        block_map[label] = {start, end};
        pos = end + 1;
    }
    for (const auto& [label, range] : block_map) {
        plan.block_start_stop.push_back(range);
        plan.block_labels.push_back(label);
    }
    return plan;
}

// replicate elements of a matrix using a vector of replicate_ids that indexs into the 
// input_matrix
MatrixT apply_replication(const MatrixT& input_matrix, const vector<int>& replicate_ids) {
    int n = replicate_ids.size();
    MatrixT expanded(n, n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            expanded(i, j) = input_matrix(replicate_ids[i], replicate_ids[j]);
    return expanded;
}

// replicates of diagonal elements of distance matrices will all be zero. This would positively
// bias similarity estimates between individuals. This function replaces replicated null diagonal 
// elements with the mean value of the within-task RDM entries.
VectorT patch_block_diagonals(const MatrixT& rdm_orig, MatrixT& rdm_full, const vector<int>& task_ids, const vector<int>& replicate_ids, const vector<int>& block_labels) {
    int dim = task_ids.size();

    for (size_t b = 0; b < block_labels.size(); ++b) {
        int label = block_labels[b];

        // find indices corresponding to current task in unexpanded RDM
        vector<int> inds;
        for (size_t i = 0; i < task_ids.size(); ++i)
            if (task_ids[i] == label) inds.push_back(i);

        // compute average distance of task conditions within-task
        REAL_T sum = static_cast<REAL_T>(0.0);
        int count = 0;
        for (int i : inds) {
            for (int j : inds) {
                if (i != j) {
                    sum += rdm_orig(i, j);
                    ++count;
                }
            }
        }
        REAL_T mean = count > 0 ? sum / count : static_cast<REAL_T>(0.0);
        
        for (int i : inds) {
            // find indices corresponding to current condition in expanded RDM
            vector<int> cond_inds;
            for (size_t j = 0; j < replicate_ids.size(); ++j)
                if (replicate_ids[j] == i) cond_inds.push_back(j);
            
            for (int k : cond_inds)
                for (int l : cond_inds)
                    if (k != l) rdm_full(k, l) = mean;
        }
    }
    for (int i = 0; i < rdm_full.rows(); ++i)
        rdm_full(i, i) = static_cast<REAL_T>(0.0);
    return matrix_to_vector(rdm_full);
}

// Helper to parse comma-separated integers from command line
vector<int> parse_task_ids(const string& arg) {
    vector<int> result;
    stringstream ss(arg);
    string token;
    while (getline(ss, token, ',')) {
        try {
            result.push_back(stoi(token));
        } catch (...) {
            throw runtime_error("Invalid task ID in input: " + token);
        }
    }
    return result;
}

/* END: Task Balancing Code */

int main(int argc, char** argv) {
    if (argc < 9) {
        cerr << "Usage: " << argv[0] << " <sub1.csv> <sub2.csv> <cov1.bin> <cov2.bin> <meta1.json> <meta2.json> <output.csv> <full_matrix:0|1> <comma_separate_task_ids>\n";
        return 1;
    }

    string subject1_file = argv[1];
    string subject2_file = argv[2];
    string cov1_file = argv[3];
    string cov2_file = argv[4];
    MetaData meta1 = parse_metadata(argv[5]);
    MetaData meta2 = parse_metadata(argv[6]);
    string out_file = argv[7];
    bool full_matrix = stoi(argv[8]) != 0;
    vector<int> task_ids;
    if (argc > 9) {
        task_ids = parse_task_ids(argv[9]);
    }

    int rdm_dim = meta1.shape_per_matrix_0;
    int n_regions = meta1.n_regions;
    REAL_T dofA = meta1.dof;
    REAL_T dofB = meta2.dof;
    size_t REAL_Ts_per_matrix = rdm_dim * (rdm_dim + 1) / 2;

    ReplicationPlan plan;

    if (task_ids.size() > 0 && task_ids.size()*(task_ids.size()-1)/2 != rdm_dim) {
        throw runtime_error("Invalid task ID length for rdm size");
    } else if (task_ids.size() > 0) {
        plan = make_replication_plan(task_ids);
    }

    MatrixT similarity_matrix;

    if (full_matrix) {
        similarity_matrix = MatrixT::Zero(n_regions, n_regions);
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < n_regions; ++i) {
            cout << "Evaluating region pair (.," << i << ")" << endl;
            vector<REAL_T> cov1 = read_region_cov_matrix(cov1_file, i, REAL_Ts_per_matrix);
            MatrixT Sigma1 = unpack_lower_triangle(cov1, rdm_dim);
            MatrixT x1 = load_csv_column(subject1_file, i, rdm_dim);
            for (int j = i; j < n_regions; ++j) {
                //cout << "Evaluating region pair (" << i << ", " << j << ")" << endl;
                vector<REAL_T> cov2 = read_region_cov_matrix(cov2_file, j, REAL_Ts_per_matrix);
                MatrixT Sigma2 = unpack_lower_triangle(cov2, rdm_dim);
                MatrixT pooled = pooled_covariance(Sigma1, dofA, Sigma2, dofB);
                MatrixT regularized = pooled + regularization_epsilon<REAL_T>() * MatrixT::Identity(rdm_dim, rdm_dim);

                // Use LAPACK for fast Cholesky factorization
                vector<REAL_T> A(regularized.size());
                Map<MatrixT> A_map(A.data(), rdm_dim, rdm_dim);
                A_map = regularized;
                int info;
                char uplo = 'L';
                LAPACK_POTRF(&uplo, &rdm_dim, A.data(), &rdm_dim, &info);
                if (info != 0) {
                    cerr << "Warning: positive definite ordered triangular refactorization (*potr) failed at region (" << j << "," << i << ") (info = " << info << ")\n";
                    similarity_matrix(i, j) = NAN;
                    if (i != j) similarity_matrix(j,i) = NAN;
                    continue;
                }

                MatrixT x2 = load_csv_column(subject2_file, j, rdm_dim);

                // Solve L * y = x using Eigen’s triangular solver
                MatrixT L = Map<MatrixT>(A.data(), rdm_dim, rdm_dim);
                L = L.triangularView<Lower>();
                MatrixT x1_whitened = L.triangularView<Lower>().solve(x1);
                MatrixT x2_whitened = L.triangularView<Lower>().solve(x2);

                // balance across tasks
                if (task_ids.size() > 0) {
                    MatrixT x1_whitened_sq = vector_to_matrix(x1_whitened.col(0), task_ids.size());
                    MatrixT x1_whitened_sq_exp = apply_replication(x1_whitened_sq, plan.replicate_ids);
                    x1_whitened = patch_block_diagonals(x1_whitened_sq, x1_whitened_sq_exp, task_ids, plan.replicate_ids, plan.block_labels);
                    MatrixT x2_whitened_sq = vector_to_matrix(x2_whitened.col(0), task_ids.size());
                    MatrixT x2_whitened_sq_exp = apply_replication(x2_whitened_sq, plan.replicate_ids);
                    x2_whitened = patch_block_diagonals(x2_whitened_sq, x2_whitened_sq_exp, task_ids, plan.replicate_ids, plan.block_labels);
                }

                similarity_matrix(i,j) = (x1_whitened.transpose() * x2_whitened)(0,0);

                REAL_T norm1 = x1_whitened.norm();
                REAL_T norm2 = x2_whitened.norm();
		        similarity_matrix(i,j) /= (norm1 * norm2);

                if (i != j) similarity_matrix(j, i) = similarity_matrix(i,j);
            }
        }
    } else {
        similarity_matrix = MatrixT::Zero(1, n_regions);
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < n_regions; ++i) {
            //cout << "Processing region " << i << endl;
            vector<REAL_T> cov1 = read_region_cov_matrix(cov1_file, i, REAL_Ts_per_matrix);
            vector<REAL_T> cov2 = read_region_cov_matrix(cov2_file, i, REAL_Ts_per_matrix);
            MatrixT Sigma1 = unpack_lower_triangle(cov1, rdm_dim);
            //save_vector_to_csv(cov1,"Sigma1.csv");
            //save_matrix_to_csv(Sigma1,"Sigma" + std::to_string(i) + ".csv");
            MatrixT Sigma2 = unpack_lower_triangle(cov2, rdm_dim);
            MatrixT pooled = pooled_covariance(Sigma1, dofA, Sigma2, dofB);
            MatrixT regularized = pooled + regularization_epsilon<REAL_T>() * MatrixT::Identity(rdm_dim, rdm_dim);

            // Use LAPACK for fast Cholesky factorization
            vector<REAL_T> A(regularized.size());
            Map<MatrixT> A_map(A.data(), rdm_dim, rdm_dim);
            A_map = regularized;
            int info;
            char uplo = 'L';
            LAPACK_POTRF(&uplo, &rdm_dim, A.data(), &rdm_dim, &info);
            if (info != 0) {
                cerr << "Warning: positive definite ordered triangular refactorization (*potrf) failed at region " << i << " (info = " << info << ")\n";
                similarity_matrix(0,i) = NAN;
                continue;
            }
            MatrixT x1 = load_csv_column(subject1_file, i, rdm_dim);
            MatrixT x2 = load_csv_column(subject2_file, i, rdm_dim);

	        // Solve L * y = x using Eigen’s triangular solver
            MatrixT L = Map<MatrixT>(A.data(), rdm_dim, rdm_dim);
            L = L.triangularView<Lower>();
            MatrixT x1_whitened = L.triangularView<Lower>().solve(x1);
            MatrixT x2_whitened = L.triangularView<Lower>().solve(x2);

            // balance across tasks
            if (task_ids.size() > 0) {
                MatrixT x1_whitened_sq = vector_to_matrix(x1_whitened.col(0), task_ids.size());
                MatrixT x1_whitened_sq_exp = apply_replication(x1_whitened_sq, plan.replicate_ids);
                x1_whitened = patch_block_diagonals(x1_whitened_sq, x1_whitened_sq_exp, task_ids, plan.replicate_ids, plan.block_labels);
                MatrixT x2_whitened_sq = vector_to_matrix(x2_whitened.col(0), task_ids.size());
                MatrixT x2_whitened_sq_exp = apply_replication(x2_whitened_sq, plan.replicate_ids);
                x2_whitened = patch_block_diagonals(x2_whitened_sq, x2_whitened_sq_exp, task_ids, plan.replicate_ids, plan.block_labels);
            }

            similarity_matrix(0,i) = (x1_whitened.transpose() * x2_whitened)(0,0);

            REAL_T norm1 = x1_whitened.norm();
            REAL_T norm2 = x2_whitened.norm();
            similarity_matrix(0,i) /= (norm1 * norm2);
        }
    }

    ofstream fout(out_file);
    if (!fout) throw runtime_error("Failed to open output file");
    for (int i = 0; i < similarity_matrix.rows(); ++i) {
        for (int j = 0; j < similarity_matrix.cols(); ++j) {
            fout << similarity_matrix(i, j);
            if (j < n_regions - 1) fout << ",";
        }
        fout << "\n";
    }

    return 0;
}

