#include <iostream>
#include <fstream>

#include "Eigen.h"

#include "VirtualSensor.h"

struct Vertex
{
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	
	// position stored as 4 floats (4th component is supposed to be 1.0)
	Vector4f position;
	// color stored as 4 unsigned char
	Vector4uc color;
};

bool WriteMesh(Vertex* vertices, unsigned int width, unsigned int height, const std::string& filename)
{
	float edgeThreshold = 0.01f; // 1cm

	// TODO 2: use the OFF file format to save the vertices grid (http://www.geomview.org/docs/html/OFF.html)
	// - have a look at the "off_sample.off" file to see how to store the vertices and triangles
	// - for debugging we recommend to first only write out the vertices (set the number of faces to zero)
	// - for simplicity write every vertex to file, even if it is not valid (position.x() == MINF) (note that all vertices in the off file have to be valid, thus, if a point is not valid write out a dummy point like (0,0,0))
	// - use a simple triangulation exploiting the grid structure (neighboring vertices build a triangle, two triangles per grid cell)
	// - you can use an arbitrary triangulation of the cells, but make sure that the triangles are consistently oriented
	// - only write triangles with valid vertices and an edge length smaller then edgeThreshold

	// TODO: Get number of vertices
	unsigned int nVertices = 0;

	// TODO: Get number of faces
	unsigned nFaces = 0;

	// Write off file
	std::ofstream outFile(filename);
	if (!outFile.is_open()) return false;

	// write header
	outFile << "COFF" << std::endl;
	
	// TODO: save vertices
	std::stringstream ss;
	ss << "# list of vertices" << std::endl;
	ss << "# X Y Z R G B A" << std::endl;
	
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			Vector4f position = vertices[y * width + x].position;
			if (position.x() == MINF) {
				position = Vector4f(0, 0, 0, 0);
			}

			Vector4uc color = vertices[y * width + x].color;
			ss << position.x() << " "
				<< position.y() << " "
				<< position.z() << " "
				<< (int)color.x() << " "
				<< (int)color.y() << " "
				<< (int)color.z() << " "
				<< (int)color.w() << std::endl;

			nVertices++;
		}
	}
	   
	// TODO: save faces
	ss << "# list of faces" << std::endl;
	ss << "# nVerticesPerFace idx0 idx1 idx2" << std::endl;

	for (int y = 0; y < height - 1; y++) {
		for (int x = 0; x < width - 1; x++) {
			int idx = y * width + x;

			Vector4f vIdx = vertices[idx].position;
			Vector4f vIdx1 = vertices[idx + 1].position;
			Vector4f vIdxW = vertices[idx + width].position;
			Vector4f vIdxW1 = vertices[idx + width + 1].position;

			if (vIdx.x() == MINF || vIdx1.x() == MINF || vIdxW.x() == MINF || vIdxW1.x() == MINF) {
				continue;
			}

			if ((vIdxW - vIdx1).squaredNorm() > edgeThreshold) {
				continue;
			}

			if ((vIdx - vIdxW).squaredNorm() <= edgeThreshold &&
				(vIdx1 - vIdx).squaredNorm() <= edgeThreshold) {
				ss << "3 " << idx << " " << idx + width << " " << idx + 1 << std::endl;
				nFaces++;
			}
			if ((vIdxW - vIdxW1).squaredNorm() <= edgeThreshold &&
				(vIdxW1 - vIdx1).squaredNorm() <= edgeThreshold) {
				ss << "3 " << idx + width << " " << idx + width + 1 << " " << idx + 1 << std::endl;
				nFaces++;
			}
		}
	}
	
	outFile << nVertices << " " << nFaces << " 0" << std::endl;
	outFile << ss.str() << std::endl;

	// close file
	outFile.close();

	return true;
}

int main()
{
	std::string filenameIn = "./data/rgbd_dataset_freiburg1_xyz/";
	std::string filenameBaseOut = "./meshes/mesh_";

	// load video
	std::cout << "Initialize virtual sensor..." << std::endl;
	VirtualSensor sensor;
	if (!sensor.Init(filenameIn))
	{
		std::cout << "Failed to initialize the sensor!\nCheck file path!" << std::endl;
		return -1;
	}

	// convert video to meshes
	while (sensor.ProcessNextFrame())
	{
		// get ptr to the current depth frame
		// depth is stored in row major (get dimensions via sensor.GetDepthImageWidth() / GetDepthImageHeight())
		float* depthMap = sensor.GetDepth();
		// get ptr to the current color frame
		// color is stored as RGBX in row major (4 byte values per pixel, get dimensions via sensor.GetColorImageWidth() / GetColorImageHeight())
		BYTE* colorMap = sensor.GetColorRGBX();

		// get depth intrinsics
		Matrix3f depthIntrinsics = sensor.GetDepthIntrinsics();
		Matrix3f depthIntrinsicsInv = sensor.GetDepthIntrinsics().inverse();
		float fovX = depthIntrinsics(0, 0);
		float fovY = depthIntrinsics(1, 1);
		float cX = depthIntrinsics(0, 2);
		float cY = depthIntrinsics(1, 2);

		// compute inverse depth extrinsics
		Matrix4f depthExtrinsicsInv = sensor.GetDepthExtrinsics().inverse();
		Matrix4f trajectory = sensor.GetTrajectory();
		Matrix4f trajectoryInv = sensor.GetTrajectory().inverse();

		// TODO 1: back-projection
		// write result to the vertices array below, keep pixel ordering!
		// if the depth value at idx is invalid (MINF) write the following values to the vertices array
		// vertices[idx].position = Vector4f(MINF, MINF, MINF, MINF);
		// vertices[idx].color = Vector4uc(0,0,0,0);
		// otherwise apply back-projection and transform the vertex to world space, use the corresponding color from the colormap
		Vertex* vertices = new Vertex[sensor.GetDepthImageWidth() * sensor.GetDepthImageHeight()];
		
		// START OWN CODE
		for (int y = 0; y < sensor.GetDepthImageHeight(); y++) {
			for (int x = 0; x < sensor.GetDepthImageWidth(); x++)
			{
				int idx = y * sensor.GetDepthImageWidth() + x;
				float depth = depthMap[idx];
				if (depth == MINF) {
					vertices[idx].position = Vector4f(MINF, MINF, MINF, MINF);
					vertices[idx].color = Vector4uc(0, 0, 0, 0);
					continue;
				}

				Vector3f p_image = Vector3f(x, y, 1) * depth;
				Vector3f p_camera = depthIntrinsicsInv * p_image;
				Vector4f p_sensor = depthExtrinsicsInv * Vector4f(p_camera[0], p_camera[1], p_camera[2], 1.0f);
				Vector4f p_world = trajectoryInv * p_sensor;

				vertices[idx].position = p_world;

				int cIdx = 4 * idx;
				vertices[idx].color = Vector4uc((unsigned char)colorMap[cIdx], (unsigned char)colorMap[cIdx + 1], (unsigned char)colorMap[cIdx + 2], (unsigned char)colorMap[cIdx + 3]);
			}
		}
		// END OWN CODE

		// write mesh file
		std::stringstream ss;
		ss << filenameBaseOut << sensor.GetCurrentFrameCnt() << ".off";
		if (!WriteMesh(vertices, sensor.GetDepthImageWidth(), sensor.GetDepthImageHeight(), ss.str()))
		{
			std::cout << "Failed to write mesh!\nCheck file path!" << std::endl;
			return -1;
		}

		// free mem
		delete[] vertices;
	}

	return 0;
}
