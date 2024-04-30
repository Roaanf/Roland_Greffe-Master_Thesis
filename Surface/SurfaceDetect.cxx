#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkCannyEdgeDetectionImageFilter.h"
#include "itkOtsuThresholdImageFilter.h"
#include "itkRescaleIntensityImageFilter.h"
#include "itkMaskImageFilter.h"
#include "itkConstantPadImageFilter.h"
#include "itkImageToVTKImageFilter.h"
#include "itkGradientImageFilter.h"
#include "itkBSplineInterpolateImageFunction.h"
#include "itkBinaryImageToLabelMapFilter.h"
#include "itkLabelMapToBinaryImageFilter.h"
#include "itkBinaryFillholeImageFilter.h"
#include "itkNormalizeImageFilter.h"
#include "itkTestingMacros.h"
#include "itkCropImageFilter.h"
#include "itkCastImageFilter.h"
#include "itkImageFileWriter.h"
#include "itkImageIterator.h"
#include <vtkActor.h>
#include <vtkNamedColors.h>
#include <vtkNew.h>
#include <vtkPolyDataMapper.h>
#include <vtkFlyingEdges3D.h>
#include <vtkSmartPointer.h>
#include <vtkBooleanOperationPolyDataFilter.h>
#include <vtkProperty.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkPCANormalEstimation.h>
#include <vtkSignedDistance.h>
#include <vtkPoissonReconstruction.h>
#include <vtkCleanPolyData.h>
#include <vtkPowerCrustSurfaceReconstruction.h>
#include <vtkWindowedSincPolyDataFilter.h>
#include <vtkSurfaceReconstructionFilter.h>
#include <vtkContourFilter.h>
#include <vtkReverseSense.h>
#include <vtkTransform.h>
#include <vtkTransformPolyDataFilter.h>
#include <vtkRenderer.h>
#include <vtkExtractSurface.h>
#include <vtkSTLReader.h>
#include <vtkSTLWriter.h>
#include <vtkCylinder.h>
#include <vtkVertexGlyphFilter.h>
#include <vtkDistancePolyDataFilter.h>
#include <vtkClipPolyData.h>
#include <vtkImplicitBoolean.h>
#include <vtkStaticPointLocator.h>
#include <vtkSurfaceNets3D.h>
#include <vtkVector.h>
#include <vtkMath.h>
#include <vtkExtractPolyDataGeometry.h>
#include <vtkConvertToPointCloud.h>
#include <vtkPolyDataWriter.h>
#include <sstream>
#include <iostream>

std::array<double,3> imageCoordToPolyDataCoord(std::array<float, 3> voxelPos, itk::Vector<double,3> imageSpacing, itk::Point<double,3> imageOrigin) {
	double coord0 = voxelPos[0] * imageSpacing[0] + imageOrigin[0];
	double coord1 = -(voxelPos[1] * imageSpacing[1] + imageOrigin[1]);
	double coord2 = voxelPos[2] * imageSpacing[2] + imageOrigin[2];
	return { coord0 ,coord1 ,coord2 };
}

std::array<int, 3> polyDataCoordToImageCoord(std::array<double, 3> polyDataPos, itk::Vector<double, 3> imageSpacing, itk::Point<double, 3> imageOrigin) {
	int coord0 = ceil((polyDataPos[0] - imageOrigin[0]) / imageSpacing[0]);
	int coord1 = ceil((-polyDataPos[1] - imageOrigin[1]) / imageSpacing[1]);
	int coord2 = ceil((polyDataPos[2] - imageOrigin[2]) / imageSpacing[2]);
	return { coord0 ,coord1 ,coord2 };
}

int main()
{
	bool writeCanny = false;
	bool writeGradient = false;
	bool boolPadding = false;
	bool cropImage = false;
	bool smoothing = false;
	bool subVoxRef = false;
	bool computePointError = true;
	enum recoAlgoEnum { ExtractSurface, Poisson, PowerCrust, SurfReconst, SurfaceNets, FlyingEdges};
	recoAlgoEnum reco = SurfaceNets;
    std::string initialMHDFilename = "rekoRolandCropped";

	typedef unsigned short InputPixelType;
	typedef itk::Image< InputPixelType, 3 > InputImageType;

	typedef unsigned char MaskPixelType;
	typedef itk::Image< MaskPixelType, 3 > MaskImageType;

	typedef float CannyPixelType;
	typedef itk::Image< CannyPixelType, 3 > CannyOutputImageType;

	InputImageType::Pointer image = itk::ReadImage<InputImageType>("./Input/"+ initialMHDFilename + ".mhd");
	InputImageType::RegionType region = image->GetLargestPossibleRegion();
	InputImageType::SizeType size = region.GetSize();

    // Will pad the image with 0 to avoid border effect when computing the gradient (only needed for mhd generated from the VTK filter)
    // Should be moved elsewhere
	if (boolPadding) {
		using PaddingFilterType = itk::ConstantPadImageFilter<InputImageType, InputImageType>;
		auto padding = PaddingFilterType::New();
		padding->SetInput(image);
		InputImageType::SizeType lowerExtendRegion;
		lowerExtendRegion.Fill(1);
		padding->SetPadLowerBound(lowerExtendRegion);
		padding->SetPadUpperBound(lowerExtendRegion);
		padding->SetConstant(0);
		padding->Update();

		typedef itk::ImageFileWriter<InputImageType> PaddedImageWriter;
		PaddedImageWriter::Pointer paddedImageWriter = PaddedImageWriter::New();
		paddedImageWriter->SetInput(padding->GetOutput());
		paddedImageWriter->SetFileName("PaddedInput.mhd");
		paddedImageWriter->Update();
		
		return true;
	}

	if (cropImage) {
		using CropImageFilterType = itk::CropImageFilter<InputImageType, InputImageType>;
		InputImageType::SizeType cropUp = {50,50,50};
		InputImageType::SizeType cropDown = { 50,0,50 };
		auto cropFilter = CropImageFilterType::New();
		cropFilter->SetInput(image);
		cropFilter->SetUpperBoundaryCropSize(cropUp);
		cropFilter->SetLowerBoundaryCropSize(cropDown);

		typedef itk::ImageFileWriter<InputImageType> CroppedImageWriter;
		CroppedImageWriter::Pointer croppedImageWriter = CroppedImageWriter::New();
		croppedImageWriter->SetInput(cropFilter->GetOutput());
		croppedImageWriter->SetFileName("CroppedInput.mhd");
		croppedImageWriter->Update();
		return true;
	}

	std::cout << size << std::endl;

	//ThresholdImage
	//marchepo en gpu ?
	typedef itk::OtsuThresholdImageFilter<InputImageType, MaskImageType> FilterType;
	FilterType::Pointer filter = FilterType::New();

	filter->SetInput(image);
	filter->SetInsideValue(0);
	filter->SetOutsideValue(1);
	filter->Update();

	MaskImageType::Pointer maskImage = filter->GetOutput();
	
	typedef itk::MaskImageFilter< InputImageType, MaskImageType, InputImageType> MaskFilterType;
	auto maskFilter = MaskFilterType::New();
	maskFilter->SetInput(image);
	maskFilter->SetMaskImage(maskImage);
	maskFilter->Update();
	std::cout << "Otsu done !" << std::endl;

	using CastFilterType = itk::CastImageFilter<InputImageType, CannyOutputImageType>;
	auto castFilter = CastFilterType::New();
	castFilter->SetInput(maskFilter->GetOutput());
	
	typedef itk::CannyEdgeDetectionImageFilter<CannyOutputImageType, CannyOutputImageType> CannyFilter;
	CannyFilter::Pointer canny = CannyFilter::New();
	canny->SetInput(castFilter->GetOutput());
	canny->SetLowerThreshold(800.0f); // 800
	canny->SetUpperThreshold(2500.0f); // 2500
	std::cout << "Threshold : " << canny->GetLowerThreshold() << " " << canny->GetUpperThreshold() << std::endl;
	canny->SetVariance(0.1);

	using RescaleFilterType = itk::RescaleIntensityImageFilter<CannyOutputImageType, MaskImageType>;
	auto rescale = RescaleFilterType::New();

	// TODO sufit d'aller chercher la gradient de base enft ... juste il me faut � la fois la direction et � la fois la norme non ?
	// faire par r�gion pour optimiser la m�moire !!!
	rescale->SetInput(canny->GetOutput());
	rescale->Update();
	MaskImageType::Pointer cannyImage = rescale->GetOutput();
	std::cout << "Canny done !" << std::endl;
	
	if (writeCanny) {
		typedef itk::ImageFileWriter<MaskImageType> ImageWriter;
		ImageWriter::Pointer imageWriter = ImageWriter::New();
		imageWriter->SetInput(cannyImage);
		imageWriter->SetFileName("TestCanny.mhd");
		imageWriter->Update();
		std::cout << "Canny write done !" << std::endl;
	}

	std::string stlFilename;
	std::string compareFileName;
	vtkNew<vtkPolyData> targetPolyData;

	if (reco == SurfaceNets || reco == FlyingEdges) {
		vtkNew<vtkSTLWriter> stlWriter;
		vtkPolyData* polyDataNotRotated;
		using FillerType = itk::BinaryFillholeImageFilter<MaskImageType>;
		auto filler = FillerType::New();
		filler->SetInput(cannyImage);
		filler->SetForegroundValue(255);
		typedef itk::ImageFileWriter<MaskImageType> ImageWriter;
		
		using ITKToVTK = itk::ImageToVTKImageFilter<MaskImageType>;
		auto ITKToVTkConverter = ITKToVTK::New();
		ITKToVTkConverter->SetInput(filler->GetOutput());
		ITKToVTkConverter->Update();
		vtkNew<vtkTransform> transP1;
		transP1->Scale(1, -1, 1);
		transP1->RotateY(90);
		vtkNew<vtkTransformPolyDataFilter> imageDataCorrect;
		/*
		ImageWriter::Pointer imageWriter = ImageWriter::New();
		imageWriter->SetInput(filler->GetOutput());
		imageWriter->SetFileName("TestFill.mhd"); // Seems good -> Not working for complex stuff :(
		imageWriter->Update();
		std::cout << "Canny fill write done !" << std::endl;
		*/
		if (reco == SurfaceNets) {
			/*
			using BinaryImageToLabelMapFilterType = itk::BinaryImageToLabelMapFilter<MaskImageType>;
			auto binaryImageToLabelMapFilter = BinaryImageToLabelMapFilterType::New();
			binaryImageToLabelMapFilter->SetInput(filler->GetOutput());
			binaryImageToLabelMapFilter->Update();
			*/
			vtkNew<vtkSurfaceNets3D> surfaceNets;
			surfaceNets->SetInputData(ITKToVTkConverter->GetOutput());
			surfaceNets->SetValue(0, 255);
			if (subVoxRef) {
				surfaceNets->SmoothingOff();
			}
			else {
				surfaceNets->SmoothingOn();
			}
			surfaceNets->SetNumberOfIterations(20);
			surfaceNets->Update();
			polyDataNotRotated = surfaceNets->GetOutput();
			stlFilename = "./Output/" + initialMHDFilename + "SurfaceNets.stl";
			compareFileName = "./Output/" + initialMHDFilename + "CompSurfaceNets.vtk";
			imageDataCorrect->SetInputData(polyDataNotRotated);
			imageDataCorrect->SetTransform(transP1);
			imageDataCorrect->Update();
			/*
			stlWriter->SetFileName(stlFilename.c_str());
			stlWriter->SetInputData(imageDataCorrect->GetOutput());
			stlWriter->Write();
			*/
			targetPolyData->DeepCopy(imageDataCorrect->GetOutput());
			
		} else {
			// No included smoothing mechanism with flyingedges
			vtkNew<vtkFlyingEdges3D> flyingEdges;
			flyingEdges->SetInputData(ITKToVTkConverter->GetOutput());
			flyingEdges->SetValue(0, 255);
			flyingEdges->ComputeNormalsOn();
			flyingEdges->ComputeGradientsOn();
			flyingEdges->Update();
			polyDataNotRotated = flyingEdges->GetOutput();
			stlFilename = "./Output/" + initialMHDFilename + "FlyingEdges.stl";
			compareFileName = "./Output/" + initialMHDFilename + "CompFlyingEdges.vtk";
			imageDataCorrect->SetInputData(polyDataNotRotated);
			imageDataCorrect->SetTransform(transP1);
			imageDataCorrect->Update();
			stlWriter->SetFileName(stlFilename.c_str());
			stlWriter->SetInputData(imageDataCorrect->GetOutput());
			stlWriter->Write();
			targetPolyData->DeepCopy(imageDataCorrect->GetOutput());
		}
		// Smoothing / BetterPrecision ?
		// Compute the Gradient Vector image
		if (subVoxRef) {
			using GradientFilterType = itk::GradientImageFilter<InputImageType, float>;
			auto gradientFilter = GradientFilterType::New();
			gradientFilter->SetInput(image);
			gradientFilter->Update();
			auto gradientImage = gradientFilter->GetOutput();

			using InitImageBSplineInterp = itk::BSplineInterpolateImageFunction<InputImageType>;
			auto bSplineInterpFilter = InitImageBSplineInterp::New();
			bSplineInterpFilter->SetSplineOrder(3);
			bSplineInterpFilter->SetInputImage(image);

			// Life hack
			vtkNew<vtkTransform> transP2;
			transP2->RotateY(90);
			vtkNew<vtkTransform> transP3;
			transP3->RotateY(-90);
			imageDataCorrect->SetInputData(targetPolyData);
			imageDataCorrect->SetTransform(transP2);
			imageDataCorrect->Update();
			vtkNew<vtkPolyData> lifeHackPart1;
			lifeHackPart1->DeepCopy(imageDataCorrect->GetOutput());
			// Direction semble bon le pb c'est l'interpolation on dirait ...
			vtkNew<vtkIdList> listOfPoints;
			double currPointCoord[3];
			itk::CovariantVector< float, 3 > currDir;
			itk::CovariantVector< float, 3 > brokenDir; // BrokenDir
			brokenDir.Fill(0);

			for (size_t iter = 0; iter < 1; iter++) {
				std::cout << "Entering loop iter " << iter << std::endl;
				int nbOfPoints = lifeHackPart1->GetNumberOfPoints();
				std::cout << nbOfPoints << std::endl;
				vtkNew<vtkPoints> newPoints;
				int modifiedPoints = 0;
				for (int i = 0; i < nbOfPoints; i++) {
					lifeHackPart1->GetPoint(i, currPointCoord);
					std::array<int, 3> imageIndex = polyDataCoordToImageCoord({ currPointCoord[0],currPointCoord[1],currPointCoord[2] }, image->GetSpacing(), image->GetOrigin());
					// gardientImageindex will have the correct position
					InputImageType::IndexType gardientImageindex{ {imageIndex[0], imageIndex[1], imageIndex[2]} };
					currDir = gradientImage->GetPixel(gardientImageindex);
					if (currDir == brokenDir) {
						newPoints->InsertNextPoint(currPointCoord[0], currPointCoord[1], currPointCoord[2]);
						continue;
					}
					currDir.Normalize();
					currDir *= image->GetSpacing()[0] * 0.2f; // The voxel size is defined by the image spacing (the 0.1f is a subvoxel refinement)
					unsigned short maxValue = 0;
					int maxIndex = 0;
					bool printInterp = false;
					if (i == 500) {
						printInterp = true;
						std::cout << "CurrDir : " << currDir << std::endl;
						std::cout << "CurrPointCoord : " << currPointCoord[0] << " " << currPointCoord[1] << " " << currPointCoord[2] << std::endl;
					}
					for (int j = -10; j < 11; j++) {
						itk::ContinuousIndex<double, 3> interpCoord = 0;
						interpCoord[0] = gardientImageindex[0] + j * currDir[0];
						interpCoord[1] = gardientImageindex[1] + j * currDir[1];
						interpCoord[2] = gardientImageindex[2] + j * currDir[2];
						double interpValue = bSplineInterpFilter->EvaluateAtContinuousIndex(interpCoord);
						if (interpValue > maxValue) {
							maxValue = interpValue;
							maxIndex = j;
						}
						if (printInterp)
							std::cout << "Interp : " << interpValue << std::endl;
					}
					std::array<double, 3> newCoords = imageCoordToPolyDataCoord({ gardientImageindex[0] + maxIndex * currDir[0],gardientImageindex[1] + maxIndex * currDir[1],gardientImageindex[2] + maxIndex * currDir[2] }, image->GetSpacing(), image->GetOrigin());
					newPoints->InsertNextPoint(newCoords[0], newCoords[1], newCoords[2]);
					if (maxIndex != 0)
						modifiedPoints++;
				}
				std::cout << "subVoxelRefinment done Mod points : " << modifiedPoints << std::endl;
				lifeHackPart1->SetPoints(newPoints);
			}

			imageDataCorrect->SetInputData(lifeHackPart1);
			imageDataCorrect->SetTransform(transP3);
			imageDataCorrect->Update();
		}
		// Smoothing the output (kindatest)
		if (smoothing && !subVoxRef) { // Makes no sense to smooth without subvox refinment
			vtkSmartPointer<vtkWindowedSincPolyDataFilter> smooth = vtkSmartPointer<vtkWindowedSincPolyDataFilter>::New();
			smooth->SetInputData(imageDataCorrect->GetOutput());
			smooth->SetPassBand(0.01);
			smooth->BoundarySmoothingOff();
			smooth->FeatureEdgeSmoothingOff();
			smooth->NonManifoldSmoothingOn();
			smooth->NormalizeCoordinatesOn();
			smooth->Update();
			targetPolyData->DeepCopy(smooth->GetOutput());
		}

		stlWriter->SetFileName(stlFilename.c_str());
		stlWriter->SetInputData(targetPolyData);
		stlWriter->Write();

	}  else {
		typedef itk::Image<float, 3> GradientImageType;

		typedef itk::GradientMagnitudeImageFilter<InputImageType, GradientImageType> GradientImageFilterType;
		GradientImageFilterType::Pointer gradientFilter = GradientImageFilterType::New();
		gradientFilter->SetInput(image);
		gradientFilter->Update();
		GradientImageType::Pointer gradientImage = gradientFilter->GetOutput();
		std::cout << "Gradient done !" << std::endl;

		if (writeGradient) {
			typedef itk::ImageFileWriter<GradientImageType> GradientWriter;
			GradientWriter::Pointer gradientWriter = GradientWriter::New();
			gradientWriter->SetInput(gradientImage);
			gradientWriter->SetFileName("TestGrad.mhd");
			gradientWriter->Update();
			std::cout << "Gradient writing done !" << std::endl;
		}

		typedef itk::Index<3> indexType;
		std::ofstream myfile;
		myfile.open("./Output/" + initialMHDFilename + "PointCloud.txt");
		for (size_t i = 0; i < size[0]; i++) {
			for (size_t j = 0; j < size[1]; j++) {
				for (size_t k = 0; k < size[2]; k++) {
					indexType currIndex;
					indexType index;
					currIndex[0] = i;
					currIndex[1] = j;
					currIndex[2] = k;
					if (cannyImage->GetPixel(currIndex) != 0.0f) {
						float xPos = 0.0f;
						float sum = 0.0f;
						float sumMul = 0.0f;
						float value;
						for (int dx = -1; dx < 2; dx++) {
							if (i + dx < 0 || i + dx >= size[0]) {
								index[0] = i;
							}
							else {
								index[0] = i + dx;
							}
							index[1] = j;
							index[2] = k;
							value = gradientImage->GetPixel(index);
							sum += value;
							sumMul += value * (dx + 1.5);
						}
						xPos = sumMul / sum;
						xPos += (static_cast<float>(i)) - 1.5;
						float yPos = 0.0f;
						sum = 0.0f;
						sumMul = 0.0f;
						for (int dy = -1; dy < 2; dy++) {
							index[0] = i;
							if (j + dy < 0 || j + dy >= size[1]) {
								index[1] = j;
							}
							else {
								index[1] = j + dy;
							}
							index[2] = k;
							value = gradientImage->GetPixel(index);
							sum += value;
							sumMul += value * (dy + 1.5);
						}
						yPos = sumMul / sum;
						yPos += (static_cast<float>(j)) - 1.5;
						float zPos = 0.0f;;
						sum = 0.0f;
						sumMul = 0.0f;
						for (int dz = -1; dz < 2; dz++) {
							index[0] = i;
							index[1] = j;
							if (k + dz < 0 || k + dz >= size[2]) {
								index[2] = k;
							}
							else {
								index[2] = k + dz;
							}
							value = gradientImage->GetPixel(index);
							sum += value;
							sumMul += value * (dz + 1.5);
						}
						zPos = sumMul / sum;
						zPos += (static_cast<float>(k)) - 1.5;

						if (xPos < 0 || xPos > size[0]) {
							continue;
						}
						if (yPos < 0 || yPos > size[1]) {
							continue;
						}
						if (zPos < 0 || zPos > size[2]) {
							continue;
						}
						if (isnan(xPos))
							continue;
						if (isnan(yPos))
							continue;
						if (isnan(zPos))
							continue;

						float absXPos = xPos * image->GetSpacing()[0] + image->GetOrigin()[0];
						float absYPos = -(yPos * image->GetSpacing()[1] + image->GetOrigin()[1]); // Pas oublier le - !!!
						float absZPos = zPos * image->GetSpacing()[2] + image->GetOrigin()[2];


						myfile << absZPos << " " << absYPos << " " << absXPos << "\n";
						//std::cout << "Position : " << xPos << " " << yPos << " " << zPos << std::endl;
					}
				}
			}
		}
		myfile.close();
		std::cout << "Point Writer done" << std::endl;

		vtkNew<vtkNamedColors> colors;
		vtkNew<vtkRenderer> renderer;
		vtkNew<vtkRenderWindow> renderWindow;
		renderWindow->AddRenderer(renderer);
		renderWindow->SetWindowName("SurfaceCreation");
		vtkNew<vtkRenderWindowInteractor> renderWindowInteractor;
		renderWindowInteractor->SetRenderWindow(renderWindow);

		std::string pointCloudFileName = "./Output/" + initialMHDFilename + "PointCloud.txt";
		std::ifstream filestream(pointCloudFileName.c_str());
		std::string line;
		vtkNew<vtkPoints> points;

		while (std::getline(filestream, line)) {
			double x, y, z;
			std::stringstream linestream;
			linestream << line;
			linestream >> x >> y >> z;
			points->InsertNextPoint(x, y, z);
		}

		filestream.close();

		vtkNew<vtkPolyData> polyData;

		polyData->SetPoints(points);

		if (computePointError) {
			// Length of edges
			vtkNew<vtkStaticPointLocator> pointLocator;
			pointLocator->SetDataSet(polyData);
			pointLocator->BuildLocator();

			// Hardcoded array because flemme
			double point1Coord[16][3] = { {-20.0, 20.0, 20.0},{-20.0, 20.0, 20.0},{-20.0, 20.0, 20.0},
										  {20.0, 20.0, 20.0},{20.0, 20.0, 20.0},{20.0, 20.0, -20.0},
										  {20.0, 20.0, -20.0},{-20.0, 20.0, -20.0},{-20.0, -20.0, -20.0},
										  {-20.0, -20.0, -20.0},{-20.0, -20.0, 20.0},{20.0, -20.0, 20.0},
										  {0.0, 50.0 ,0.0},{0.0, 50.0 ,0.0},{0.0, 50.0 ,0.0},{0.0, 50.0 ,0.0} };

			double point2Coord[16][3] = { {20.0, 20.0, 20.0},{-20.0, -20.0, 20.0},{-20.0, 20.0, -20.0},
										  {20.0, 20.0, -20.0},{20.0, -20.0, 20.0},{20.0, -20.0, -20.0},
										  {-20.0, 20.0, -20.0},{-20.0, -20.0, -20.0},{-20.0, -20.0, 20.0},
										  {20.0, -20.0, -20.0},{20.0, -20.0, 20.0},{20.0, -20.0, -20.0},
										  {-20.0, 20.0, 20.0},{-20.0, 20.0, -20.0},{20.0, 20.0, -20.0},{20.0, 20.0, 20.0} };
			double distancesError[16];
			// Loop through all 16 outer edges
			for (size_t i = 0; i < 16; i++) {
				vtkIdType point1ID = pointLocator->FindClosestPoint(point1Coord[i]);
				double* point1Point = polyData->GetPoint(point1ID);
				double point1[3] = { point1Point[0], point1Point[1], point1Point[2] };
				vtkIdType point2ID = pointLocator->FindClosestPoint(point2Coord[i]);
				double* point2Point = polyData->GetPoint(point2ID);
				double point2[3] = { point2Point[0], point2Point[1], point2Point[2] };
				double distance = std::sqrt(vtkMath::Distance2BetweenPoints(point1, point2));
				//compute distance btw the two points
				if (i < 12) {
					distancesError[i] = abs(distance - 40.0);
				}
				else {
					distancesError[i] = abs(distance - 41.2309);
				}
				std::cout << distancesError[i] << " ";
			}
			std::cout << std::endl;
			// Angles
			double anglesErrors[3]; // Voir si j'en fais plus ?
			double realAngles[3] = { 90.0, 90.0, 43.313 };
			double point1Vec[4][3] = { {20.0, -20.0, -20.0},{-20.0, -20.0, -20.0},{-20.0, -20.0, 20.0},
										  {-20.0, 20.0, 20.0} };
			double point2Vec[4][3] = { {-20.0, -20.0, 20.0},{-20.0, -20.0, 20.0},{-20.0, 20.0, 20.0},
										  {0.0, 50.0, 0.0} };
			// Flemme de la boucle polalalalala
			std::vector<vtkVector3d> vectors;
			for (size_t i = 0; i < 4; i++) {
				vtkIdType point1ID = pointLocator->FindClosestPoint(point1Vec[i]);
				double* point1Point = polyData->GetPoint(point1ID);
				double point1[3] = { point1Point[0], point1Point[1], point1Point[2] };
				vtkIdType point2ID = pointLocator->FindClosestPoint(point2Vec[i]);
				double* point2Point = polyData->GetPoint(point2ID);
				double point2[3] = { point2Point[0], point2Point[1], point2Point[2] };
				double diff[3] = { point2[0] - point1[0], point2[1] - point1[1], point2[2] - point1[2] };
				//create vector
				vtkVector3d vector = vtkVector3d(diff);
				vectors.push_back(vector);
			}

			anglesErrors[0] = vtkMath::DegreesFromRadians(vtkMath::AngleBetweenVectors(vectors[0].GetData(), vectors[2].GetData())) - realAngles[0];
			anglesErrors[1] = vtkMath::DegreesFromRadians(vtkMath::AngleBetweenVectors(vectors[1].GetData(), vectors[2].GetData())) - realAngles[1];
			anglesErrors[2] = vtkMath::DegreesFromRadians(vtkMath::AngleBetweenVectors(vectors[3].GetData(), vectors[2].GetData())) - realAngles[2];

			std::cout << anglesErrors[0] << " " << anglesErrors[1] << " " << anglesErrors[2] << std::endl;

			vtkNew<vtkCylinder> cylinder;
			cylinder->SetCenter(-20.0, 0.0, 0.0);
			cylinder->SetRadius(10.15);
			cylinder->SetAxis(1.0, 0.0, 0.0);
			vtkNew<vtkImplicitBoolean> boolean;
			boolean->AddFunction(cylinder);
			vtkNew<vtkExtractPolyDataGeometry> extractPolyDataGeometry;
			extractPolyDataGeometry->SetInputData(polyData);
			extractPolyDataGeometry->SetExtractInside(true);
			extractPolyDataGeometry->SetImplicitFunction(boolean);
			extractPolyDataGeometry->Update();

			vtkNew<vtkConvertToPointCloud> pcConvert;
			pcConvert->SetInputConnection(extractPolyDataGeometry->GetOutputPort());
			pcConvert->SetCellGenerationMode(vtkConvertToPointCloud::NO_CELLS);
			pcConvert->Update();
			vtkPolyData* cylPoints = pcConvert->GetOutput();
			std::ofstream myfile;
			myfile.open("./Output/" + initialMHDFilename + "CylinderPointCloud.txt");
			for (int i = 0; i < cylPoints->GetNumberOfPoints(); i++) {
				double* point = cylPoints->GetPoint(i);
				myfile << point[0] << " " << point[1] << " " << point[2] << "\n";
			}
			myfile.close();
			vtkNew<vtkPolyDataWriter> vtkWriter;
			std::string cylCropFilname = "./Output/" + initialMHDFilename + "TestCylinderCrop.vtk";
			vtkWriter->SetFileName(cylCropFilname.c_str());
			vtkWriter->SetInputConnection(extractPolyDataGeometry->GetOutputPort());
			vtkWriter->Write();
		}

		vtkNew<vtkSignedDistance> distance;
		vtkNew<vtkPCANormalEstimation> normals;
		vtkNew<vtkVertexGlyphFilter> glyphFilter;
		vtkNew<vtkSTLWriter> stlWriter;
		int sampleSize = 15;
		normals->SetInputData(polyData);
		normals->SetSampleSize(sampleSize);
		normals->SetNormalOrientationToGraphTraversal();
		normals->FlipNormalsOn();
		distance->SetInputConnection(normals->GetOutputPort());

		switch (reco) {
			case ExtractSurface: {
				double bounds[6];
				polyData->GetBounds(bounds);
				double range[3];
				for (int i = 0; i < 3; ++i)
				{
					range[i] = bounds[2 * i + 1] - bounds[2 * i];
				}
				glyphFilter->SetInputData(polyData);
				glyphFilter->Update();

				int dimension = 512;
				double radius;
				radius = std::max(std::max(range[0], range[1]), range[2]) / static_cast<double>(dimension) * 4; // ~4 voxels

				distance->SetRadius(radius);
				distance->SetDimensions(dimension, dimension, dimension);
				distance->SetBounds(bounds[0] - range[0] * .1, bounds[1] + range[0] * .1,
					bounds[2] - range[1] * .1, bounds[3] + range[1] * .1,
					bounds[4] - range[2] * .1, bounds[5] + range[2] * .1);
				vtkNew<vtkExtractSurface> surfaceExtract;
				surfaceExtract->SetInputConnection(distance->GetOutputPort());
				surfaceExtract->SetRadius(radius * .99);
				surfaceExtract->Update();
				targetPolyData->DeepCopy(surfaceExtract->GetOutput());
				stlFilename = "./Output/" + initialMHDFilename + "ExtractSurface.stl";
				compareFileName = "./Output/" + initialMHDFilename + "CompExtractSurface.vtk";
				stlWriter->SetFileName(stlFilename.c_str());
				stlWriter->SetInputConnection(surfaceExtract->GetOutputPort());
				stlWriter->Write();
				break;
			}
			case Poisson: {
				vtkSmartPointer<vtkPoissonReconstruction> surfacePois = vtkSmartPointer<vtkPoissonReconstruction>::New();
				surfacePois->SetDepth(12);
				surfacePois->SetInputConnection(normals->GetOutputPort());
				surfacePois->Update();
				targetPolyData->DeepCopy(surfacePois->GetOutput());
				stlFilename = "./Output/" + initialMHDFilename + "Poisson.stl";
				compareFileName = "./Output/" + initialMHDFilename + "CompPoisson.vtk";
				stlWriter->SetFileName(stlFilename.c_str());
				stlWriter->SetInputConnection(surfacePois->GetOutputPort());
				stlWriter->Write();
				break;
			}
			case PowerCrust: {
				vtkSmartPointer<vtkPowerCrustSurfaceReconstruction> surfacePowerCrust = vtkSmartPointer<vtkPowerCrustSurfaceReconstruction>::New();
				surfacePowerCrust->SetInputData(polyData);
				stlFilename = "./Output/" + initialMHDFilename + "PowerCrust.stl";
				compareFileName = "./Output/" + initialMHDFilename + "CompPowerCrust.vtk";
				surfacePowerCrust->Update();
				targetPolyData->DeepCopy(surfacePowerCrust->GetOutput());
				stlWriter->SetFileName(stlFilename.c_str());
				stlWriter->SetInputConnection(surfacePowerCrust->GetOutputPort());
				stlWriter->Write();
				break;
			}
			case SurfReconst: {
				vtkNew<vtkSurfaceReconstructionFilter> surf;
				surf->SetInputData(polyData);
				surf->SetNeighborhoodSize(10);
				surf->SetSampleSpacing(0.1);

				vtkNew<vtkContourFilter> contourFilter;
				contourFilter->SetInputConnection(surf->GetOutputPort());
				contourFilter->SetValue(0, 0.0);

				// Sometimes the contouring algorithm can create a volume whose gradient
				// vector and ordering of polygon (using the right hand rule) are
				// inconsistent. vtkReverseSense cures this problem.
				vtkNew<vtkReverseSense> surface;
				surface->SetInputConnection(contourFilter->GetOutputPort());
				surface->ReverseCellsOn();
				surface->ReverseNormalsOn();
				surface->Update();
				targetPolyData->DeepCopy(surface->GetOutput());
				stlFilename = "./Output/" + initialMHDFilename + "SurfaceExtractFilter.stl";
				compareFileName = "./Output/" + initialMHDFilename + "CompSurfaceExtractFilter.vtk";
				stlWriter->SetFileName(stlFilename.c_str());
				stlWriter->SetInputConnection(surface->GetOutputPort());
				stlWriter->Write();
				break;
			}
		}
	}

	//Load reference STL
	vtkNew<vtkSTLReader> referenceReader;
	referenceReader->SetFileName("./Input/InitialModel.stl");
	referenceReader->Update();
	vtkNew<vtkCleanPolyData> referencePolyData;
	referencePolyData->SetInputData(referenceReader->GetOutput());
	std::cout << "Reference Loaded" << std::endl;

	vtkNew<vtkCleanPolyData> targetCleanPolyData;
	targetCleanPolyData->SetInputData(targetPolyData);

	vtkNew<vtkDistancePolyDataFilter> distanceFilter;
	distanceFilter->SetInputConnection(1, referencePolyData->GetOutputPort());
	distanceFilter->SetInputConnection(0, targetCleanPolyData->GetOutputPort());
	distanceFilter->SetSignedDistance(false);
	distanceFilter->Update();

	vtkNew<vtkPolyDataWriter> vtkWriter;
	
	vtkWriter->SetFileName(compareFileName.c_str());
	vtkWriter->SetInputConnection(distanceFilter->GetOutputPort());
	vtkWriter->Write();

	if (computePointError) {
		// Length of edges
		vtkNew<vtkStaticPointLocator> pointLocator;
		vtkPolyData* polyData = targetPolyData;
		pointLocator->SetDataSet(polyData);
		pointLocator->BuildLocator();

		// Hardcoded array because flemme
		double point1Coord[16][3] = { {-20.0, 20.0, 20.0},{-20.0, 20.0, 20.0},{-20.0, 20.0, 20.0},
									  {20.0, 20.0, 20.0},{20.0, 20.0, 20.0},{20.0, 20.0, -20.0},
									  {20.0, 20.0, -20.0},{-20.0, 20.0, -20.0},{-20.0, -20.0, -20.0},
									  {-20.0, -20.0, -20.0},{-20.0, -20.0, 20.0},{20.0, -20.0, 20.0},
									  {0.0, 50.0 ,0.0},{0.0, 50.0 ,0.0},{0.0, 50.0 ,0.0},{0.0, 50.0 ,0.0} };

		double point2Coord[16][3] = { {20.0, 20.0, 20.0},{-20.0, -20.0, 20.0},{-20.0, 20.0, -20.0},
									  {20.0, 20.0, -20.0},{20.0, -20.0, 20.0},{20.0, -20.0, -20.0},
									  {-20.0, 20.0, -20.0},{-20.0, -20.0, -20.0},{-20.0, -20.0, 20.0},
									  {20.0, -20.0, -20.0},{20.0, -20.0, 20.0},{20.0, -20.0, -20.0},
									  {-20.0, 20.0, 20.0},{-20.0, 20.0, -20.0},{20.0, 20.0, -20.0},{20.0, 20.0, 20.0} };
		double distancesError[16];
		// Loop through all 16 outer edges
		for (size_t i = 0; i < 16; i++) {
			vtkIdType point1ID = pointLocator->FindClosestPoint(point1Coord[i]);
			double* point1Point = polyData->GetPoint(point1ID);
			double point1[3] = { point1Point[0], point1Point[1], point1Point[2] };
			vtkIdType point2ID = pointLocator->FindClosestPoint(point2Coord[i]);
			double* point2Point = polyData->GetPoint(point2ID);
			double point2[3] = { point2Point[0], point2Point[1], point2Point[2] };
			double distance = std::sqrt(vtkMath::Distance2BetweenPoints(point1, point2));
			//compute distance btw the two points
			if (i < 12) {
				distancesError[i] = abs(distance - 40.0);
			}
			else {
				distancesError[i] = abs(distance - 41.2309);
			}
			std::cout << distancesError[i] << " ";
		}
		std::cout << std::endl;
		// Angles
		double anglesErrors[3]; // Voir si j'en fais plus ?
		double realAngles[3] = { 90.0, 90.0, 43.313 };
		double point1Vec[4][3] = { {20.0, -20.0, -20.0},{-20.0, -20.0, -20.0},{-20.0, -20.0, 20.0},
									  {-20.0, 20.0, 20.0} };
		double point2Vec[4][3] = { {-20.0, -20.0, 20.0},{-20.0, -20.0, 20.0},{-20.0, 20.0, 20.0},
									  {0.0, 50.0, 0.0} };
		// Flemme de la boucle polalalalala
		std::vector<vtkVector3d> vectors;
		for (size_t i = 0; i < 4; i++) {
			vtkIdType point1ID = pointLocator->FindClosestPoint(point1Vec[i]);
			double* point1Point = polyData->GetPoint(point1ID);
			double point1[3] = { point1Point[0], point1Point[1], point1Point[2] };
			vtkIdType point2ID = pointLocator->FindClosestPoint(point2Vec[i]);
			double* point2Point = polyData->GetPoint(point2ID);
			double point2[3] = { point2Point[0], point2Point[1], point2Point[2] };
			double diff[3] = { point2[0] - point1[0], point2[1] - point1[1], point2[2] - point1[2] };
			//create vector
			vtkVector3d vector = vtkVector3d(diff);
			vectors.push_back(vector);
		}

		anglesErrors[0] = vtkMath::DegreesFromRadians(vtkMath::AngleBetweenVectors(vectors[0].GetData(), vectors[2].GetData())) - realAngles[0];
		anglesErrors[1] = vtkMath::DegreesFromRadians(vtkMath::AngleBetweenVectors(vectors[1].GetData(), vectors[2].GetData())) - realAngles[1];
		anglesErrors[2] = vtkMath::DegreesFromRadians(vtkMath::AngleBetweenVectors(vectors[3].GetData(), vectors[2].GetData())) - realAngles[2];

		std::cout << anglesErrors[0] << " " << anglesErrors[1] << " " << anglesErrors[2] << std::endl;
		vtkNew<vtkCylinder> cylinder;
		cylinder->SetCenter(-20.0, 0.0, 0.0);
		cylinder->SetRadius(10.15);
		cylinder->SetAxis(1.0, 0.0, 0.0);
		vtkNew<vtkImplicitBoolean> boolean;
		boolean->AddFunction(cylinder);
		vtkNew<vtkExtractPolyDataGeometry> extractPolyDataGeometry;
		extractPolyDataGeometry->SetInputData(polyData);
		extractPolyDataGeometry->SetExtractInside(true);
		extractPolyDataGeometry->SetImplicitFunction(boolean);
		extractPolyDataGeometry->Update();

		vtkNew<vtkConvertToPointCloud> pcConvert;
		pcConvert->SetInputConnection(extractPolyDataGeometry->GetOutputPort());
		pcConvert->SetCellGenerationMode(vtkConvertToPointCloud::NO_CELLS);
		pcConvert->Update();
		vtkPolyData* cylPoints = pcConvert->GetOutput();
		std::ofstream myfile;
		myfile.open("PointCloud.txt");
		for (int i = 0; i < cylPoints->GetNumberOfPoints(); i++) {
			double* point = cylPoints->GetPoint(i);
			myfile << point[0] << " " << point[1] << " " << point[2] << "\n";
		}
		myfile.close();
		vtkNew<vtkPolyDataWriter> vtkWriter;
		vtkWriter->SetFileName("./Output/TestCylinderCrop.vtk");
		vtkWriter->SetInputConnection(extractPolyDataGeometry->GetOutputPort());
		vtkWriter->Write();

		// Do the fitting -> https://www.geometrictools.com/Documentation/LeastSquaresFitting.pdf (banger)
		// Flemme de le coder je vais compiler GeometricTools à la place UwU

	}

	return EXIT_SUCCESS;
}