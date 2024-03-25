#include "itkMesh.h"
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkCannyEdgeDetectionImageFilter.h"
#include "itkMeshFileWriter.h"
#include "itkOtsuThresholdImageFilter.h"
#include "itkRescaleIntensityImageFilter.h"
#include "itkTestingMacros.h"
#include "itkCastImageFilter.h"
#include "itkImageFileWriter.h"
#include <iostream>

int main()
{
	typedef itk::Image< unsigned short, 3 > InputImageType;
	typedef itk::Image< unsigned char, 3 > MaskImageType;
	typedef itk::Image< float, 3 > CannyOutputImageType;
	InputImageType::Pointer image = itk::ReadImage<InputImageType>("./Data/fdk.mhd");

	InputImageType::RegionType region = image->GetLargestPossibleRegion();
	InputImageType::SizeType size = region.GetSize();
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

	using CastFilterType = itk::CastImageFilter<InputImageType, CannyOutputImageType>;
	auto castFilter = CastFilterType::New();
	castFilter->SetInput(maskFilter->GetOutput());
	
	
	typedef itk::CannyEdgeDetectionImageFilter<CannyOutputImageType, CannyOutputImageType> CannyFilter;
	CannyFilter::Pointer canny = CannyFilter::New();
	canny->SetInput(castFilter->GetOutput());
	canny->SetLowerThreshold(100.0f); // Presque parfait banger
	canny->SetUpperThreshold(500.f);
	std::cout << "Threshold : " << canny->GetLowerThreshold() << " " << canny->GetUpperThreshold() << std::endl;
	canny->SetVariance(0.1);
	
	typedef itk::ImageFileWriter<CannyOutputImageType> ImageWriter;
	ImageWriter::Pointer imageWriter = ImageWriter::New();
	imageWriter->SetInput(canny->GetOutput());
	imageWriter->SetFileName("TestCanny.mhd");
	imageWriter->Update();
	
	std::cout << "Writer done" << std::endl;

	return EXIT_SUCCESS;
}