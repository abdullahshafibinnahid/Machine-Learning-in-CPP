// ============================================================================
//  ml_notebook.cpp
//
//  A tiny, dependency-free "ML notebook" in C++.
//
//  Problem : predict house price (price_k, in thousands) from
//            area_sqft, bedrooms, age_years
//  Model   : multivariate linear regression, trained with batch
//            gradient descent (features are standardized first)
//  Files   : train.csv            -> has the target column, used to fit the model
//            test.csv             -> no target column, this is what we predict
//            sample_submission.csv-> shows the expected submission format
//            submission.csv       -> OUTPUT: our real predictions, written by this program
//
//  Build   : g++ -O2 -std=c++17 ml_notebook.cpp -o ml_notebook
//  Run     : ./ml_notebook
// ============================================================================

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <iomanip>
#include <stdexcept>

// ---------------------------------------------------------------------------
// A minimal CSV reader. Good enough for simple, comma-separated numeric data
// with a header row (no quoted commas, which we don't need here).
// ---------------------------------------------------------------------------
struct CsvTable {
    std::vector<std::string> header;
    std::vector<std::vector<std::string>> rows;
};

static CsvTable read_csv(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + path);
    }

    CsvTable table;
    std::string line;

    auto strip_cr = [](std::string s) {
        if (!s.empty() && s.back() == '\r') s.pop_back();
        return s;
    };

    // header
    if (std::getline(file, line)) {
        line = strip_cr(line);
        std::stringstream ss(line);
        std::string cell;
        while (std::getline(ss, cell, ',')) table.header.push_back(cell);
    }

    // rows
    while (std::getline(file, line)) {
        line = strip_cr(line);
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string cell;
        std::vector<std::string> row;
        while (std::getline(ss, cell, ',')) row.push_back(cell);
        table.rows.push_back(row);
    }
    return table;
}

static int col_index(const CsvTable& t, const std::string& name) {
    for (size_t i = 0; i < t.header.size(); ++i)
        if (t.header[i] == name) return static_cast<int>(i);
    throw std::runtime_error("Column not found: " + name);
}

// ---------------------------------------------------------------------------
// Linear Regression via batch gradient descent, with z-score normalization
// of the input features (helps gradient descent converge quickly/stably).
// ---------------------------------------------------------------------------
class LinearRegression {
public:
    void fit(const std::vector<std::vector<double>>& X,
             const std::vector<double>& y,
             double lr = 0.05,
             int epochs = 5000) {
        n_features_ = X[0].size();
        compute_normalization(X);

        std::vector<std::vector<double>> Xn = normalize(X);

        weights_.assign(n_features_, 0.0);
        bias_ = 0.0;

        size_t m = Xn.size();
        for (int epoch = 0; epoch < epochs; ++epoch) {
            std::vector<double> grad_w(n_features_, 0.0);
            double grad_b = 0.0;

            for (size_t i = 0; i < m; ++i) {
                double pred = predict_normalized(Xn[i]);
                double error = pred - y[i];
                for (size_t j = 0; j < n_features_; ++j)
                    grad_w[j] += error * Xn[i][j];
                grad_b += error;
            }

            for (size_t j = 0; j < n_features_; ++j)
                weights_[j] -= lr * (grad_w[j] / m);
            bias_ -= lr * (grad_b / m);

            if ((epoch + 1) % 1000 == 0) {
                double mse = 0.0;
                for (size_t i = 0; i < m; ++i) {
                    double e = predict_normalized(Xn[i]) - y[i];
                    mse += e * e;
                }
                mse /= m;
                std::cout << "  epoch " << (epoch + 1)
                          << "  train RMSE = " << std::fixed
                          << std::setprecision(3) << std::sqrt(mse) << "\n";
            }
        }
    }

    double predict(const std::vector<double>& raw_features) const {
        std::vector<double> x = normalize_one(raw_features);
        return predict_normalized(x);
    }

    void print_equation(const std::vector<std::string>& feature_names) const {
        std::cout << "\nLearned model (in original feature units):\n  price_k = ";
        // Convert normalized-space weights back to raw-space so the printed
        // equation is directly interpretable.
        double raw_bias = bias_;
        std::cout << std::fixed << std::setprecision(4);
        for (size_t j = 0; j < n_features_; ++j) {
            double raw_w = weights_[j] / std_[j];
            raw_bias -= raw_w * mean_[j];
            std::cout << (raw_w >= 0 ? "+ " : "- ") << std::abs(raw_w)
                      << " * " << feature_names[j] << " ";
        }
        std::cout << (raw_bias >= 0 ? "+ " : "- ") << std::abs(raw_bias) << "\n";
    }

private:
    size_t n_features_ = 0;
    std::vector<double> weights_;
    double bias_ = 0.0;
    std::vector<double> mean_, std_;

    void compute_normalization(const std::vector<std::vector<double>>& X) {
        mean_.assign(n_features_, 0.0);
        std_.assign(n_features_, 0.0);
        size_t m = X.size();

        for (size_t j = 0; j < n_features_; ++j) {
            double sum = 0.0;
            for (size_t i = 0; i < m; ++i) sum += X[i][j];
            mean_[j] = sum / m;
        }
        for (size_t j = 0; j < n_features_; ++j) {
            double sq = 0.0;
            for (size_t i = 0; i < m; ++i) {
                double d = X[i][j] - mean_[j];
                sq += d * d;
            }
            double variance = sq / m;
            std_[j] = (variance > 1e-12) ? std::sqrt(variance) : 1.0;
        }
    }

    std::vector<std::vector<double>> normalize(const std::vector<std::vector<double>>& X) const {
        std::vector<std::vector<double>> Xn(X.size(), std::vector<double>(n_features_));
        for (size_t i = 0; i < X.size(); ++i)
            Xn[i] = normalize_one(X[i]);
        return Xn;
    }

    std::vector<double> normalize_one(const std::vector<double>& x) const {
        std::vector<double> out(n_features_);
        for (size_t j = 0; j < n_features_; ++j)
            out[j] = (x[j] - mean_[j]) / std_[j];
        return out;
    }

    double predict_normalized(const std::vector<double>& xn) const {
        double s = bias_;
        for (size_t j = 0; j < n_features_; ++j) s += weights_[j] * xn[j];
        return s;
    }
};

int main() {
    std::cout << "==================================================\n";
    std::cout << " C++ ML Notebook: House Price Prediction (Linear Regression)\n";
    std::cout << "==================================================\n\n";

    // ---- 1. Load training data -------------------------------------------
    std::cout << "[1/4] Loading train.csv ...\n";
    CsvTable train = read_csv("train.csv");

    int c_area   = col_index(train, "area_sqft");
    int c_beds   = col_index(train, "bedrooms");
    int c_age    = col_index(train, "age_years");
    int c_price  = col_index(train, "price_k");

    std::vector<std::vector<double>> X_train;
    std::vector<double> y_train;
    X_train.reserve(train.rows.size());
    y_train.reserve(train.rows.size());

    for (auto& row : train.rows) {
        X_train.push_back({
            std::stod(row[c_area]),
            std::stod(row[c_beds]),
            std::stod(row[c_age])
        });
        y_train.push_back(std::stod(row[c_price]));
    }
    std::cout << "  loaded " << X_train.size() << " training rows\n\n";

    // ---- 2. Train the model ----------------------------------------------
    std::cout << "[2/4] Training linear regression (gradient descent) ...\n";
    LinearRegression model;
    model.fit(X_train, y_train, /*lr=*/0.05, /*epochs=*/5000);
    model.print_equation({"area_sqft", "bedrooms", "age_years"});
    std::cout << "\n";

    // ---- 3. Load test data and predict ------------------------------------
    std::cout << "[3/4] Loading test.csv and generating predictions ...\n";
    CsvTable test = read_csv("test.csv");
    int t_id   = col_index(test, "id");
    int t_area = col_index(test, "area_sqft");
    int t_beds = col_index(test, "bedrooms");
    int t_age  = col_index(test, "age_years");

    std::vector<std::pair<std::string, double>> predictions;
    for (auto& row : test.rows) {
        std::vector<double> feats = {
            std::stod(row[t_area]),
            std::stod(row[t_beds]),
            std::stod(row[t_age])
        };
        double pred = model.predict(feats);
        predictions.push_back({row[t_id], pred});
    }
    std::cout << "  predicted " << predictions.size() << " test rows\n\n";

    // ---- 4. Write submission.csv ------------------------------------------
    std::cout << "[4/4] Writing submission.csv ...\n";
    std::ofstream out("submission.csv");
    out << "id,price_k\n";
    out << std::fixed << std::setprecision(2);
    for (auto& [id, pred] : predictions) {
        out << id << "," << pred << "\n";
    }
    out.close();

    std::cout << "  wrote submission.csv with " << predictions.size() << " rows\n\n";
    std::cout << "Done. Compare submission.csv against sample_submission.csv\n";
    std::cout << "(same id column / format, but with real predicted values).\n";

    return 0;
}
