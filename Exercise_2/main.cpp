#include <iostream>

#include "Eigen.h"
#include "ImplicitSurface.h"
#include "Volume.h"
#include "MarchingCubes.h"

int main()
{
	std::string filenameIn = "./data/normalized.pcb";
	std::string filenameOut = "result.off";

	// implicit surface
	ImplicitSurface* surface;
	// TODO: you have to switch between these surface types
	//surface = new Sphere(Eigen::Vector3d(0.5, 0.5, 0.5), 0.4);
	//surface = new Torus(Eigen::Vector3d(0.5, 0.5, 0.5), 0.4, 0.1);
	//surface = new Hoppe(filenameIn);
	surface = new RBF(filenameIn);

	// fill volume with signed distance values
	unsigned int mc_res = 50; // resolution of the grid, for debugging you can reduce the resolution (-> faster)
	double mc_res_cube = mc_res * mc_res *mc_res;
	double lastPrint = 0;
	Volume vol(Vector3d(-0.1,-0.1,-0.1), Vector3d(1.1,1.1,1.1), mc_res, mc_res, mc_res, 1);
	for (unsigned int x = 0; x < vol.getDimX(); x++)
	{
		for (unsigned int y = 0; y < vol.getDimY(); y++)
		{
			for (unsigned int z = 0; z < vol.getDimZ(); z++)
			{
				double currentPrint = (x * mc_res * mc_res + y * mc_res + z) / mc_res_cube;
				if (currentPrint - lastPrint >= 0.01) {
					std::cerr << "Fill volume with signed distance values: " << std::round(currentPrint * 100 + 0.5) << "%" << std::endl;
					lastPrint = currentPrint;
				}
				Eigen::Vector3d p = vol.pos(x, y, z);
				double val = surface->Eval(p);
				vol.set(x, y, z, val);
			}
		}
	}

	// extract the zero iso-surface using marching cubes
	SimpleMesh mesh;
	for (unsigned int x = 0; x < vol.getDimX() - 1; x++)
	{
		std::cerr << "Marching Cubes on slice " << x << " of " << vol.getDimX() << std::endl;

		for (unsigned int y = 0; y < vol.getDimY() - 1; y++)
		{
			for (unsigned int z = 0; z < vol.getDimZ() - 1; z++)
			{
				ProcessVolumeCell(&vol, x, y, z, 0.00f, &mesh);
			}
		}
	}

	// write mesh to file
	if (!mesh.WriteMesh(filenameOut))
	{
		std::cout << "ERROR: unable to write output file!" << std::endl;
		return -1;
	}

	delete surface;

	return 0;
}
