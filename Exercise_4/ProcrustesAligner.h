#pragma once
#include "SimpleMesh.h"

class ProcrustesAligner {
public:
	Matrix4f estimatePose(const std::vector<Vector3f>& sourcePoints, const std::vector<Vector3f>& targetPoints) {
		ASSERT(sourcePoints.size() == targetPoints.size() && "The number of source and target points should be the same, since every source point is matched with corresponding target point.");

		// We estimate the pose between source and target points using Procrustes algorithm.
		// Our shapes have the same scale, therefore we don't estimate scale. We estimated rotation and translation
		// from source points to target points.

		auto sourceMean = computeMean(sourcePoints);
		auto targetMean = computeMean(targetPoints);
		
		Matrix3f rotation = estimateRotation(sourcePoints, sourceMean, targetPoints, targetMean);
		Vector3f translation = computeTranslation(sourceMean, targetMean);

		// To apply the pose to point x on shape X in the case of Procrustes, we execute:
		// 1. Translation of a point to the shape Y: x' = x + t
		// 2. Rotation of the point around the mean of shape Y: 
		//    y = R (x' - yMean) + yMean = R (x + t - yMean) + yMean = R x + (R t - R yMean + yMean)
		
		Matrix4f estimatedPose = Matrix4f::Identity();
		estimatedPose.block(0, 0, 3, 3) = rotation;
		estimatedPose.block(0, 3, 3, 1) = rotation * translation - rotation * targetMean + targetMean;

		return estimatedPose;
	}

private:
	Vector3f computeMean(const std::vector<Vector3f>& points) {
		Vector3f mean = Vector3f::Zero();
		
		for (auto point : points) {
			mean += point;
		}

		return mean / points.size();
	}

	Matrix3f estimateRotation(const std::vector<Vector3f>& sourcePoints, const Vector3f& sourceMean, const std::vector<Vector3f>& targetPoints, const Vector3f& targetMean) {
		// TODO: Estimate the rotation from source to target points, following the Procrustes algorithm.
		// To compute the singular value decomposition you can use JacobiSVD() from Eigen.
		// Important: The covariance matrices should contain mean-centered source/target points.

		std::vector<Vector3f> meanCenteredSourcePoints = sourcePoints;
		std::vector<Vector3f> meanCenteredTargetPoints = targetPoints;

		std::transform(meanCenteredSourcePoints.begin(), meanCenteredSourcePoints.end(), meanCenteredSourcePoints.begin(), [sourceMean](Vector3f point) -> Vector3f{ return point - sourceMean; });
		std::transform(meanCenteredTargetPoints.begin(), meanCenteredTargetPoints.end(), meanCenteredTargetPoints.begin(), [targetMean](Vector3f point) -> Vector3f { return point - targetMean; });

		Eigen::Matrix3f A = Eigen::Matrix3f();
		A << sourcePoints[3] - sourcePoints[0], sourcePoints[2] - sourcePoints[0], sourcePoints[1] - sourcePoints[0];
		Eigen::Matrix3f B = Eigen::Matrix3f();
		B << targetPoints[3] - targetPoints[0], targetPoints[2] - targetPoints[0], targetPoints[1] - targetPoints[0];

		Eigen::JacobiSVD<Matrix3f> svd = Eigen::JacobiSVD<Matrix3f>(B * A.transpose(), Eigen::ComputeFullU | Eigen::ComputeFullV);

		return svd.matrixU() * svd.matrixV().transpose();
	}

	Vector3f computeTranslation(const Vector3f& sourceMean, const Vector3f& targetMean) {
		// TODO: Compute the translation vector from source to target points.
		
		return targetMean - sourceMean;
	}
};