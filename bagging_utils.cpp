#include "bagging_utils.hpp"

#include <queue>
#include <random>
#include <limits>
#include <algorithm>
#include <iostream>


namespace BaggingUtils {
/*
    Implements the Reservoir Sampling algorithm to draw a reservoir sample of
    size k, without replacement.
 */
std::vector<int> reservoir_sample ( const int k, const std::vector<int>& S ) {
    // the reservoir sample container
    std::vector<int> R(k);
    // misc for random number generation
    std::random_device r;
    std::mt19937 gen(r());
    std::uniform_int_distribution<> dis;

    // fill the reservoir sequentially with the first k items in S[0,...,k-1]
    for(decltype(R.size()) i = 0; i < R.size(); ++i) {
        R[i] = S[i];
    }

    // replace the elements with gradually decreasing probability
    for(decltype(S.size()) i = k; i < S.size(); ++i) {
        dis = std::uniform_int_distribution<>(0, i);
        decltype(i) j = dis(gen);
        if (j < k) {
            R[j] = S[i];
        }
    }

    return R;
}

std::vector<int> sampling_without_replacement ( const int k, std::vector<int> S ) {
    std::random_device rd;
    std::mt19937 generator(rd());


    int max = S.size() - 1;

    std::vector<int> result;

    for(int i = 1; i <= k; i++) {
        std::uniform_int_distribution<> uniform_dist(0, max);
        int index = uniform_dist(generator);
        std::swap(S[index], S[max]);
        result.push_back(S[max]);
        max--;
        if(max == 0) max = S.size() - 1;
    }

    return result;
}

/*
    Splits data into a training and a validation set.
 */
sampled_data split_data ( const Matrix& data_in, const double s_v,
    const bool preserve_class_balance )
{
    const int n_attr = data_in.n_cols - 1;
    sampled_data result;

    if(preserve_class_balance) {
        // process positive and negative instances separately to keep balance
        std ::vector<int> pos_indices, neg_indices;
        for(int i = 0; i < data_in.n_rows; ++i) {
            if(data_in(i, n_attr) < 0)
                neg_indices.push_back(i);
            else
                pos_indices.push_back(i);
        }

        // now, draw a reservoir sample from both the positive and negative sets
        const int pos_valid_size = s_v * pos_indices.size();
        const int neg_valid_size = s_v * neg_indices.size();
        const int pos_train_size = pos_indices.size() - pos_valid_size;
        const int neg_train_size = neg_indices.size() - neg_valid_size;
        auto pos_sample_indices = sampling_without_replacement(pos_valid_size, pos_indices);
        auto neg_sample_indices = sampling_without_replacement(neg_valid_size, neg_indices);
        std::sort(pos_sample_indices.begin(), pos_sample_indices.end());
        std::sort(neg_sample_indices.begin(), neg_sample_indices.end());

        // used for the trainig set
        std::vector<int> inv_pos_sample_indices, inv_neg_sample_indices;

        for(auto ind : pos_indices) {
            if(!std::binary_search(pos_sample_indices.begin(),
                pos_sample_indices.end(), ind))
                inv_pos_sample_indices.push_back(ind);
        }

        for(auto ind : neg_indices) {
            if(!std::binary_search(neg_sample_indices.begin(),
                neg_sample_indices.end(), ind))

            inv_neg_sample_indices.push_back(ind);
        }

        // construct the validation and training sets
        int i_pos_t = 0, i_neg_t = 0, i_pos_v = 0, i_neg_v = 0;
        result.x_train.push_back(Matrix(neg_train_size, n_attr));
        result.x_train.push_back(Matrix(pos_train_size, n_attr));
        result.y_train.push_back(Vector(neg_train_size));
        result.y_train.push_back(Vector(pos_train_size));
        result.x_valid.push_back(Matrix(neg_valid_size, n_attr));
        result.x_valid.push_back(Matrix(pos_valid_size, n_attr));
        result.y_valid.push_back(Vector(neg_valid_size));
        result.y_valid.push_back(Vector(pos_valid_size));

        for(auto pos_ind : pos_sample_indices) {
            result.x_valid[1].row(i_pos_v) = data_in.row(pos_ind).head(n_attr);
            result.y_valid[1](i_pos_v) = data_in(pos_ind, n_attr);
            ++i_pos_v;
        }
        for(auto neg_ind : neg_sample_indices) {
            result.x_valid[0].row(i_neg_v) = data_in.row(neg_ind).head(n_attr);
            result.y_valid[0](i_neg_v) = data_in(neg_ind, n_attr);
            ++i_neg_v;
        }
        for(auto pos_ind : inv_pos_sample_indices) {
            result.x_train[1].row(i_pos_t) = data_in.row(pos_ind).head(n_attr);
            result.y_train[1](i_pos_t) = data_in(pos_ind, n_attr);
            ++i_pos_t;
        }
        for(auto neg_ind : inv_neg_sample_indices) {
            result.x_train[0].row(i_neg_t) = data_in.row(neg_ind).head(n_attr);
            result.y_train[0](i_neg_t) = data_in(neg_ind, n_attr);
            ++i_neg_t;
        }

    }
    else {
        // inv_samlple_indices is used for the training set
        std::vector<int> population, inv_sample_indices;
        const int valid_size = (int) (s_v * data_in.n_rows);
        const int train_size = data_in.n_rows - valid_size;
        int i_v = 0, i_t = 0;
        result.x_train.push_back(Matrix(train_size, n_attr));
        result.y_train.push_back(Vector(train_size));
        result.x_valid.push_back(Matrix(valid_size, n_attr));
        result.y_valid.push_back(Vector(valid_size));

        for(int i = 0; i < data_in.n_rows; ++i) {
            population.push_back(i);
        }

        auto sample_indices = sampling_without_replacement(valid_size, population);
        std::sort(sample_indices.begin(), sample_indices.end());

        for(int i = 0; i < data_in.n_rows; ++i) {
            if(!std::binary_search(sample_indices.begin(), sample_indices.end(), i)) {
                inv_sample_indices.push_back(i);
            }
        }

        for(auto ind : sample_indices) {
            result.x_valid[0].row(i_v) = data_in.row(ind).head(n_attr);
            result.y_valid[0](i_v) = (double)(data_in(ind, n_attr));
            ++i_v;
        }

        for(auto ind : inv_sample_indices) {
            result.x_train[0].row(i_t) = data_in.row(ind).head(n_attr);
            result.y_train[0](i_t) = (double)(data_in(ind, n_attr));
            ++i_t;
        }
    }

    return result;
}

/*
    randomized BFS needed for the Ford-Fulkerson algorithm
*/
int bfs ( std_matrix& residual_graph, int source, int sink ) {
    std::vector<bool> visited(residual_graph.size(), false);
    std::vector<int> from(residual_graph.size(), -1);

    std::queue<int> q;
    q.push(source);
    visited[source] = true;
    from[source] = -1;

    bool flag = true;
    while(!q.empty() && flag) {
        int u = q.front();
        q.pop();

        std::vector<int> neighbors;
        for(int v = 0; v < residual_graph[u].size(); ++v)
            if(visited[v] == false && residual_graph[u][v] > 0)
                neighbors.push_back(v);

        std::random_shuffle(neighbors.begin(), neighbors.end());

        for(auto v : neighbors) {
            q.push(v);
            visited[v] = true;
            from[v] = u;
            if(v == sink) {
                flag = false;
                break;
            }
        }
    }

    // find the capacity (min) of the shortest path from source to sink
    int prev, curr = sink, path_capacity = std::numeric_limits<int>::max();
    while(from[curr] > -1) {
        prev = from[curr];
        path_capacity = std::min(path_capacity, residual_graph[prev][curr]);
        curr = prev;
    }

    // update flow along the path
    curr = sink;
    while(from[curr] > -1) {
        prev = from[curr];
        residual_graph[prev][curr] -= path_capacity;
        residual_graph[curr][prev] += path_capacity;
        curr = prev;
    }

    if(path_capacity == std::numeric_limits<int>::max()) return 0;
    else return path_capacity;
}

/*
    Implements the Ford-Fulkerson algorithm for Maximum Network Flow Problem
    NOTE: modifies the passed graph in-place (rGraph is the residual graph)
*/
int max_flow ( std_matrix& graph, int source, int sink, std_matrix& residual_result ) {

    int result = 0;

    std_matrix residual_graph(graph.size());
    for(decltype(graph.size()) i = 0; i < graph.size(); ++i) {
        residual_graph[i] = std::vector<int>(graph[i].size());
        for(decltype(graph[i].size()) j = 0; j < graph[i].size(); ++j)
            residual_graph[i][j] = graph[i][j];
    }

    while(true) {
        int path_capacity = bfs(residual_graph, source, sink);
        if(path_capacity == 0) break;
        else result += path_capacity;
    }

    residual_result = residual_graph;

    return result;
}

}   // end namespace
