#include <vtkActor.h>
#include <vtkNamedColors.h>
#include <vtkNew.h>
#include <vtkPolyDataMapper.h>
#include <vtkSmartPointer.h>
#include <vtkBooleanOperationPolyDataFilter.h>
#include <vtkProperty.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>
#include <vtkSTLReader.h>
#include <vtkVertexGlyphFilter.h>
#include <sstream>

int main(int argc, char* argv[])
{
    vtkNew<vtkNamedColors> colors;

    std::vector<std::string> files;
    std::string box = "./Data/box.stl";
    std::string cylinder = "./Data/cylinder.stl";
    std::string pyramide = "./Data/pyramide.stl";
    files.push_back(box);
    files.push_back(cylinder);
    files.push_back(pyramide);

    std::vector<vtkSmartPointer<vtkPolyData>> polyDataVec;

    vtkNew<vtkRenderer> renderer;
    vtkNew<vtkRenderWindow> renderWindow;
    renderWindow->AddRenderer(renderer);
    renderWindow->SetWindowName("ReadSTL");
    vtkNew<vtkRenderWindowInteractor> renderWindowInteractor;
    renderWindowInteractor->SetRenderWindow(renderWindow);

    for (const std::string &i : files) {
        vtkNew<vtkSTLReader> reader;
        reader->SetFileName(i.c_str());
        reader->Update();
        auto poly_data = vtkSmartPointer<vtkPolyData>::New();
        poly_data->ShallowCopy(reader->GetOutput());
        polyDataVec.push_back(poly_data);
        /*
        // Visualize
        vtkNew<vtkPolyDataMapper> mapper;
        mapper->SetInputConnection(reader->GetOutputPort());

        vtkNew<vtkActor> actor;
        actor->SetMapper(mapper);
        actor->GetProperty()->SetDiffuse(0.8);
        actor->GetProperty()->SetDiffuseColor(
            colors->GetColor3d("LightSteelBlue").GetData());
        actor->GetProperty()->SetSpecular(0.3);
        actor->GetProperty()->SetSpecularPower(60.0);
        renderer->AddActor(actor);
        */
    }
    auto addFilter = vtkSmartPointer<vtkBooleanOperationPolyDataFilter>::New();
    addFilter->SetInputData(0, polyDataVec[0]);
    addFilter->SetInputData(1, polyDataVec[2]);
    addFilter->SetOperationToUnion();
    addFilter->Update();
    
    auto diffFilter = vtkSmartPointer<vtkBooleanOperationPolyDataFilter>::New();
    diffFilter->SetInputData(0, addFilter->GetOutput());
    diffFilter->SetInputData(1, polyDataVec[1]);
    diffFilter->SetOperationToDifference();
    diffFilter->Update();

    vtkNew<vtkPolyDataMapper> mapper;
    mapper->SetInputConnection(diffFilter->GetOutputPort());

    vtkNew<vtkActor> actor;
    actor->SetMapper(mapper);
    actor->GetProperty()->SetDiffuse(0.8);
    actor->GetProperty()->SetDiffuseColor(colors->GetColor3d("LightSteelBlue").GetData());
    actor->GetProperty()->SetSpecular(0.3);
    actor->GetProperty()->SetSpecularPower(60.0);
    renderer->AddActor(actor);

    std::string filename = "./Data/PointCloud.txt";
    std::ifstream filestream(filename.c_str());
    std::string line;
    vtkNew<vtkPoints> points;

    while (std::getline(filestream, line)){
        double x, y, z;
        std::stringstream linestream;
        linestream << line;
        linestream >> x >> y >> z;

        points->InsertNextPoint(x, y, z);
    }

    filestream.close();

    vtkNew<vtkPolyData> polyData;

    polyData->SetPoints(points);

    vtkNew<vtkVertexGlyphFilter> glyphFilter;
    glyphFilter->SetInputData(polyData);
    glyphFilter->Update();

    vtkNew<vtkPolyDataMapper> mapperPoint;
    mapperPoint->SetInputConnection(glyphFilter->GetOutputPort());
    vtkNew<vtkActor> actorPoint;
    actorPoint->SetMapper(mapperPoint);
    actorPoint->GetProperty()->SetColor(colors->GetColor3d("Red").GetData());

    renderer->AddActor(actorPoint);

    renderer->SetBackground(colors->GetColor3d("Gray").GetData());
    renderWindow->Render();
    renderWindowInteractor->Start();

    return EXIT_SUCCESS;
}