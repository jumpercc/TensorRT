#include "BatchStream.h"
#include "EntropyCalibrator.h"
#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"

#include "NvInfer.h"
#include "NvUffParser.h"
#include <cuda_runtime_api.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <unistd.h>

using Severity = nvinfer1::ILogger::Severity;

class TRT_Logger : public nvinfer1::ILogger
{
    nvinfer1::ILogger::Severity _verbosity;
    std::ostream* _ostream;

public:
    TRT_Logger(Severity verbosity = Severity::kWARNING, std::ostream& ostream = std::cerr)
        : _verbosity(verbosity)
        , _ostream(&ostream)
    {
    }
    void log(Severity severity, const char* msg) override
    {
        if (severity <= _verbosity)
        {
            time_t rawtime = std::time(0);
            char buf[256];
            strftime(&buf[0], 256, "%Y-%m-%d %H:%M:%S", std::gmtime(&rawtime));
            const char* sevstr = (severity == Severity::kINTERNAL_ERROR ? "    BUG"
                : severity == Severity::kERROR                          ? "  ERROR"
                : severity == Severity::kWARNING                        ? "WARNING"
                : severity == Severity::kINFO                           ? "   INFO"
                                                                        : "UNKNOWN");
            (*_ostream) << "[" << buf << " " << sevstr << "] " << msg << std::endl;
        }
    }
};

struct UffSSDParams
{
    std::string uffFileName = "ssd_relu6.uff";
    std::string labelsFileName = "ssd_coco_labels.txt";
    int outputClsSize = 91;
    float visualThreshold = 0.5;
    std::vector<std::string> inputTensorNames = {"Input"};
    std::vector<std::string> outputTensorNames = {"NMS", "NMS_1"};

    void SetNetFilesDirectory(std::string net_files_directory)
    {
        uffFileName = net_files_directory + "/" + uffFileName;
        labelsFileName = net_files_directory + "/" + labelsFileName;
    }
};

class UffSSD
{
    template <typename T>
    using SampleUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;

public:
    UffSSD()
        : mEngine(nullptr)
    {
    }

    //!
    //! \brief Function builds the network engine
    //!
    bool build(std::string net_files_directory);

    //!
    //! \brief Runs the TensorRT inference engine for this sample
    //!
    std::vector<std::vector<float>> infer(uint8_t* rgb_data);

    //!
    //! \brief Cleans up any state created in the sample class
    //!
    bool teardown();

    //!
    //! \brief returns H, W for input image
    //!
    std::pair<int, int> get_input_image_size()
    {
        return {mInputDims.d[1], mInputDims.d[2]};
    }

private:
    UffSSDParams mParams; //!< The parameters for the sample.

    nvinfer1::Dims mInputDims; //!< The dimensions of the input to the network.

    std::shared_ptr<nvinfer1::ICudaEngine> mEngine; //!< The TensorRT engine used to run the network

    //!
    //! \brief Parses an UFF model for SSD and creates a TensorRT network
    //!
    bool constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
        SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
        SampleUniquePtr<nvuffparser::IUffParser>& parser);

    //!
    //! \brief Reads the input and mean data, preprocesses, and stores the result in a managed buffer
    //!
    bool processInput(const samplesCommon::BufferManager& buffers, const uint8_t* rgb_data);

    //!
    //! \brief Filters output detections and verify results
    //!
    bool verifyOutput(const samplesCommon::BufferManager& buffers);
};

//!
//! \brief Creates the network, configures the builder and creates the network engine
//!
//! \details This function creates the SSD network by parsing the UFF model and builds
//!          the engine that will be used to run SSD (mEngine)
//!
//! \return Returns true if the engine was created successfully and false otherwise
//!
bool UffSSD::build(std::string net_files_directory)
{
    TRT_Logger logger;
    initLibNvInferPlugins(&logger, "");

    mParams.SetNetFilesDirectory(net_files_directory);

    auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger));
    if (!builder)
    {
        return false;
    }

    auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetwork());
    if (!network)
    {
        return false;
    }

    auto config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
    {
        return false;
    }

    auto parser = SampleUniquePtr<nvuffparser::IUffParser>(nvuffparser::createUffParser());
    if (!parser)
    {
        return false;
    }

    auto constructed = constructNetwork(builder, network, config, parser);
    if (!constructed)
    {
        return false;
    }

    assert(network->getNbInputs() == 1);
    mInputDims = network->getInput(0)->getDimensions();
    assert(mInputDims.nbDims == 3);

    assert(network->getNbOutputs() == 2);

    return true;
}

//!
//! \brief Uses a UFF parser to create the SSD Network and marks the
//!        output layers
//!
//! \param network Pointer to the network that will be populated with the SSD network
//!
//! \param builder Pointer to the engine builder
//!
bool UffSSD::constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
    SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
    SampleUniquePtr<nvuffparser::IUffParser>& parser)
{
    parser->registerInput(mParams.inputTensorNames[0].c_str(), DimsCHW(3, 300, 300), nvuffparser::UffInputOrder::kNCHW);
    parser->registerOutput(mParams.outputTensorNames[0].c_str());

    auto parsed = parser->parse(mParams.uffFileName.c_str(), *network, DataType::kFLOAT);
    if (!parsed)
    {
        return false;
    }

    builder->setMaxBatchSize(1);
    config->setMaxWorkspaceSize(1_GiB);

    // Calibrator life time needs to last until after the engine is built.
    std::unique_ptr<IInt8Calibrator> calibrator;

    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
        builder->buildEngineWithConfig(*network, *config), samplesCommon::InferDeleter());
    if (!mEngine)
    {
        return false;
    }

    return true;
}

//!
//! \brief Runs the TensorRT inference engine for this sample
//!
//! \details This function is the main execution function of the sample. It allocates the buffer,
//!          sets inputs and executes the engine.
//!
std::vector<std::vector<float>> UffSSD::infer(uint8_t* rgb_data)
{
    // Create RAII buffer manager object
    samplesCommon::BufferManager buffers(mEngine, 1);

    auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    if (!context)
    {
        return {};
    }

    // Read the input data into the managed buffers
    assert(mParams.inputTensorNames.size() == 1);
    if (!processInput(buffers, rgb_data))
    {
        return {};
    }

    // Memcpy from host input buffers to device input buffers
    buffers.copyInputToDevice();

    bool status = context->execute(1, buffers.getDeviceBindings().data());
    if (!status)
    {
        return {};
    }

    // Memcpy from device output buffers to host output buffers
    buffers.copyOutputToHost();

    // Post-process detections
    const float visualThreshold = mParams.visualThreshold;
    const int outputClsSize = mParams.outputClsSize;

    const float* detectionOut = static_cast<const float*>(buffers.getHostBuffer(mParams.outputTensorNames[0]));
    const int* keepCount = static_cast<const int*>(buffers.getHostBuffer(mParams.outputTensorNames[1]));

    std::vector<std::vector<float>> result;
    for (int i = 0; i < keepCount[0]; ++i)
    {
        const float* det = &detectionOut[0] + i * 7;
        if (det[2] < visualThreshold)
        {
            continue;
        }

        assert(int(det[1]) < outputClsSize);
        result.push_back({
            det[1],         // class number
            det[2],         // confidence
            det[3], det[4], // x0, y0 (0-1)
            det[5], det[6], // x1, y1 (0-1)
        });
    }

    return result;
}

//!
//! \brief Cleans up any state created in the sample class
//!
bool UffSSD::teardown()
{
    //! Clean up the libprotobuf files as the parsing is complete
    //! \note It is not safe to use any other part of the protocol buffers library after
    //! ShutdownProtobufLibrary() has been called.
    nvuffparser::shutdownProtobufLibrary();
    return true;
}

//!
//! \brief Reads the input and mean data, preprocesses, and stores the result in a managed buffer
//!
bool UffSSD::processInput(const samplesCommon::BufferManager& buffers, const uint8_t* rgb_data)
{
    const int inputC = mInputDims.d[0];
    const int inputH = mInputDims.d[1];
    const int inputW = mInputDims.d[2];

    float* hostDataBuffer = static_cast<float*>(buffers.getHostBuffer(mParams.inputTensorNames[0]));
    // Host memory for input buffer
    for (int c = 0; c < inputC; ++c)
    {
        // The color image to input should be in BGR order
        for (unsigned j = 0, volChl = inputH * inputW; j < volChl; ++j)
        {
            hostDataBuffer[c * volChl + j] = (2.0 / 255.0) * float(rgb_data[j * inputC + c]) - 1.0;
        }
    }

    return true;
}

UffSSD* a_net;

void StartupDetector(std::string net_files_directory)
{
    if (a_net == nullptr)
    {
        a_net = new UffSSD();
        if (!a_net->build(net_files_directory))
        {
            throw std::runtime_error("build failed");
        }
    }
}

std::vector<std::vector<float>> DetectObjects(uint8_t* rgb_data)
{
    if (a_net == nullptr)
    {
        throw std::runtime_error("you should call StartupDetector first");
    }

    return a_net->infer(rgb_data);
}

void CleanupDetector()
{
    if (a_net != nullptr)
    {
        a_net->teardown();
        delete a_net;
        a_net = nullptr;
    }
}
