#include "itkMesh.h"
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkCannyEdgeDetectionImageFilter.h"
#include "itkMeshFileWriter.h"
#include "itkOtsuThresholdImageFilter.h"
#include "itkRescaleIntensityImageFilter.h"
#include "itkMaskImageFilter.h"
#include "itkConstantPadImageFilter.h"
#include "itkGradientImageFilter.h"
#include "itkBSplineInterpolateImageFunction.h"
#include "itkBinaryImageToLabelMapFilter.h"
#include "itkLabelMapToBinaryImageFilter.h"
#include "itkNormalizeImageFilter.h"
#include "itkTestingMacros.h"
#include "itkCastImageFilter.h"
#include "itkImageFileWriter.h"
#include "itkImageIterator.h"
#include <iostream>

int main()
{
	bool writeCanny = true;
	bool writeGradient = true;
	bool boolPadding = false;

	typedef unsigned char InputPixelType;
	typedef itk::Image< InputPixelType, 3 > InputImageType;

	typedef unsigned char MaskPixelType;
	typedef itk::Image< MaskPixelType, 3 > MaskImageType;

	typedef float CannyPixelType;
	typedef itk::Image< CannyPixelType, 3 > CannyOutputImageType;

	std::string filename = "PaddedInput";

	InputImageType::Pointer image = itk::ReadImage<InputImageType>("./Data/"+ filename + ".mhd");
	InputImageType::RegionType region = image->GetLargestPossibleRegion();
	InputImageType::SizeType size = region.GetSize();

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
	canny->SetLowerThreshold(0.1f);
	canny->SetUpperThreshold(1.0f);
	std::cout << "Threshold : " << canny->GetLowerThreshold() << " " << canny->GetUpperThreshold() << std::endl;
	canny->SetVariance(0.1);

	using RescaleFilterType = itk::RescaleIntensityImageFilter<CannyOutputImageType, MaskImageType>;
	auto rescale = RescaleFilterType::New();

	// TODO sufit d'aller chercher la gradient de base enft ... juste il me faut à la fois la direction et à la fois la norme non ?
	// faire par région pour optimiser la mémoire !!!
	rescale->SetInput(canny->GetOutput());
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

	/*
	typedef itk::BinaryImageToLabelMapFilter<MaskImageType> BinaryMapToLabelType;
	BinaryMapToLabelType::Pointer labelCreator = BinaryMapToLabelType::New();

	labelCreator->SetInput(cannyImage);
	labelCreator->Update();

	unsigned int max = 0;
	unsigned int maxPosition = 0;
	std::vector<unsigned long> labelsToRemove;

	
	for (unsigned int i = 0; i < labelCreator->GetOutput()->GetNumberOfLabelObjects(); ++i)
	{
		// Get the ith region
		BinaryMapToLabelType::OutputImageType::LabelObjectType* labelObject = labelCreator->GetOutput()->GetNthLabelObject(i);
		if (labelObject->Size() > max) {
			max = labelObject->Size();
			maxPosition = i;
		}
	}

	for (unsigned int i = 0; i < labelCreator->GetOutput()->GetNumberOfLabelObjects(); ++i)
	{
		// Get the ith region
		if (i != maxPosition) {
			BinaryMapToLabelType::OutputImageType::LabelObjectType* labelObject = labelCreator->GetOutput()->GetNthLabelObject(i);
			labelsToRemove.push_back(labelObject->GetLabel());
		}
	}

	for (unsigned long i : labelsToRemove){
		labelCreator->GetOutput()->RemoveLabel(i);
	}
	

	using LabelMapToBinaryImageFilterType = itk::LabelMapToBinaryImageFilter<BinaryMapToLabelType::OutputImageType, MaskImageType>;
	auto labelMapToBinaryImageFilter = LabelMapToBinaryImageFilterType::New();
	labelMapToBinaryImageFilter->SetInput(labelCreator->GetOutput());
	labelMapToBinaryImageFilter->Update();

	ImageWriter::Pointer imageWriterLabel = ImageWriter::New();
	imageWriterLabel->SetInput(labelMapToBinaryImageFilter->GetOutput());
	imageWriterLabel->SetFileName("TestCannyLabel.mhd");
	imageWriterLabel->Update();

	return 0;
	*/
	typedef itk::Image<float, 3> GradientImageType;

	typedef itk::GradientMagnitudeImageFilter<InputImageType, GradientImageType> GradientImageFilterType;
	GradientImageFilterType::Pointer gradientFilter = GradientImageFilterType::New();
	gradientFilter->SetInput(image);
	gradientFilter->Update();
	GradientImageType::Pointer gradientImage = (gradientFilter->GetOutput());
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
	/*
	while (!cannyIt.IsAtEnd()) {
		if (cannyIt.Get() != 0.0f) {

			float xPos;
			float sum = 0.0f;
			float sumMul = 0.0f;
			for (int dx = -1; dx < 2; dx++) {
				sum += gradientImage->GetPixel();
				sumMul += gradientImage->GetBufferPointer()[i + dx, j, k] * (dx + 1.5);
			}
			xPos = sumMul / sum;
			xPos += (static_cast<float>(i)) - 1.5;
			float yPos;
			sum = 0.0f;
			sumMul = 0.0f;
			for (int dy = -1; dy < 2; dy++) {
				sum += gradientImage->GetBufferPointer()[i, j + dy, k];
				sumMul += gradientImage->GetBufferPointer()[i, j + dy, k] * (dy + 1.5);
			}
			yPos = sumMul / sum;
			yPos += (static_cast<float>(j)) - 1.5;
			float zPos;
			sum = 0.0f;
			sumMul = 0.0f;
			for (int dz = -1; dz < 2; dz++) {
				sum += gradientImage->GetBufferPointer()[i + dz, j, k];
				sumMul += gradientImage->GetBufferPointer()[i + dz, j, k] * (dz + 1.5);
			}
			zPos = sumMul / sum;
			zPos += (static_cast<float>(i)) - 1.5;

			MeshType::PointType point;
			point[0] = xPos;
			point[1] = yPos;
			point[2] = zPos;

			mesh->SetPoint(mesh->GetNumberOfPoints(), point);
		}
		++cannyIt;
		++gradientIt;
	}
	*/
	// Should be done with an iterator in a ideal world ... (but the example doesn't work ...)
	//CannyPixelType* cannyBuffer = cannyImage->GetBufferPointer(); // Bad should do it correctly !!!
	//GradientType* gradientBuffer = gradientImage->GetBufferPointer(); // Bad should do it correctly !!!
	// Crash here when changing the size
	std::ofstream myfile;
	myfile.open("PointCloud.txt");
	for (size_t i = 0; i < size[0]; i++) {
		for (size_t j = 0; j < size[1]; j++){
			for (size_t k = 0; k < size[2]; k++){
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
					float absYPos = -(yPos * image->GetSpacing()[1] + image->GetOrigin()[1]);
					float absZPos = zPos * image->GetSpacing()[2] + image->GetOrigin()[2];


					myfile << absZPos << " " << absYPos << " " << absXPos << "\n";
					//std::cout << "Position : " << xPos << " " << yPos << " " << zPos << std::endl;
				}
			}
		}
	}
	myfile.close();
	std::cout << "Writer done" << std::endl;

	/*
	MeshWriterType::Pointer meshWriter = MeshWriterType::New();
	meshWriter->SetInput(mesh);
	meshWriter->SetFileName("TestMesh.vtk");
	meshWriter->Update();

	typedef itk::ImageFileWriter<GradientImageType> ImageWriter;
	ImageWriter::Pointer imageWriter = ImageWriter::New();
	imageWriter->SetInput(gradientFilter->GetOutput());
	imageWriter->SetFileName("TestCanny.mhd");
	imageWriter->Update();
	*/
	

	return EXIT_SUCCESS;
}