#include <opencv2/core.hpp>
#include <vector>

// Decomposição simples da matriz essencial
void decomposeEssential(const cv::Mat& E, cv::Mat& R1, cv::Mat& R2, cv::Mat& t) {
    cv::SVD svd(E, cv::SVD::MODIFY_A);
    cv::Mat W = (cv::Mat_<double>(3,3) << 0,-1,0, 1,0,0, 0,0,1);

    R1 = svd.u * W * svd.vt;
    R2 = svd.u * W.t() * svd.vt;
    t = svd.u.col(2);
    // Garantir que R1 e R2 são rotações válidas
    if (cv::determinant(R1) < 0) R1 = -R1;
    if (cv::determinant(R2) < 0) R2 = -R2;
}

// Triangulação linear simples para um par de pontos
cv::Mat triangulateSimple(const cv::Mat& P1, const cv::Mat& P2, const cv::Point2f& pt1, const cv::Point2f& pt2) {
    cv::Mat A(4, 4, CV_64F);
    A.row(0) = pt1.x * P1.row(2) - P1.row(0);
    A.row(1) = pt1.y * P1.row(2) - P1.row(1);
    A.row(2) = pt2.x * P2.row(2) - P2.row(0);
    A.row(3) = pt2.y * P2.row(2) - P2.row(1);

    cv::SVD svd(A, cv::SVD::MODIFY_A);
    cv::Mat X = svd.vt.row(3).t();
    return X / X.at<double>(3,0);
}

// Função principal simplificada
void simpleRecoverPose(const cv::Mat& E, const std::vector<cv::Point2f>& points1,
                      const std::vector<cv::Point2f>& points2, const cv::Mat& K,
                      cv::Mat& R, cv::Mat& t) {
    cv::Mat R1, R2, t_;
    decomposeEssential(E, R1, R2, t_);

    // Projeções
    cv::Mat P0 = cv::Mat::eye(3, 4, CV_64F);
    std::vector<cv::Mat> P = {
        (cv::Mat_<double>(3,4) << R1.at<double>(0,0),R1.at<double>(0,1),R1.at<double>(0,2), t_.at<double>(0),
                                  R1.at<double>(1,0),R1.at<double>(1,1),R1.at<double>(1,2), t_.at<double>(1),
                                  R1.at<double>(2,0),R1.at<double>(2,1),R1.at<double>(2,2), t_.at<double>(2)),
        (cv::Mat_<double>(3,4) << R2.at<double>(0,0),R2.at<double>(0,1),R2.at<double>(0,2), t_.at<double>(0),
                                  R2.at<double>(1,0),R2.at<double>(1,1),R2.at<double>(1,2), t_.at<double>(1),
                                  R2.at<double>(2,0),R2.at<double>(2,1),R2.at<double>(2,2), t_.at<double>(2)),
        (cv::Mat_<double>(3,4) << R1.at<double>(0,0),R1.at<double>(0,1),R1.at<double>(0,2), -t_.at<double>(0),
                                  R1.at<double>(1,0),R1.at<double>(1,1),R1.at<double>(1,2), -t_.at<double>(1),
                                  R1.at<double>(2,0),R1.at<double>(2,1),R1.at<double>(2,2), -t_.at<double>(2)),
        (cv::Mat_<double>(3,4) << R2.at<double>(0,0),R2.at<double>(0,1),R2.at<double>(0,2), -t_.at<double>(0),
                                  R2.at<double>(1,0),R2.at<double>(1,1),R2.at<double>(1,2), -t_.at<double>(1),
                                  R2.at<double>(2,0),R2.at<double>(2,1),R2.at<double>(2,2), -t_.at<double>(2))
    };

    cv::Mat KP0 = K * P0;
    std::vector<cv::Mat> KP;
    for (int i = 0; i < 4; ++i) {
        KP.push_back(K * P[i]);
    }

    int best = 0, maxFront = -1;
    for (int i = 0; i < 4; ++i) {
        int front = 0;
        for (size_t j = 0; j < points1.size(); ++j) {
            cv::Mat X = triangulateSimple(KP0, KP[i], points1[j], points2[j]);
            double z1 = X.at<double>(2);
            cv::Mat X2 = KP[i] * X;
            double z2 = X2.at<double>(2);
            if (z1 > 0 && z2 > 0) front++;
        }
        if (front > maxFront) {
            maxFront = front;
            best = i;
        }
    }

    // Retorna a melhor solução
    if (best == 0) { R = R1; t = t_; }
    else if (best == 1) { R = R2; t = t_; }
    else if (best == 2) { R = R1; t = -t_; }
    else { R = R2; t = -t_; }
}