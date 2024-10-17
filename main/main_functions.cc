#include "main_functions.h"

#include "detection_responder.h"
#include "image_provider.h"
#include "model_settings.h"
#include "person_detect_model_data.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/micro_profiler.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include <esp_heap_caps.h>
#include <esp_timer.h>
#include <esp_log.h>
#include "esp_main.h"
#include "esp_task_wdt.h"
#include "driver/gpio.h"

// Globals, used for compatibility with Arduino-style sketches.
namespace {
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;

// In order to use optimized tensorflow lite kernels, a signed int8_t quantized
// model is preferred over the legacy unsigned model format. This means that
// throughout this project, input images must be converted from unisgned to
// signed format. The easiest and quickest way to convert from unsigned to
// signed 8-bit integers is to subtract 128 from the unsigned value to get a
// signed value.
#ifdef CONFIG_IDF_TARGET_ESP32S3
constexpr int scratchBufSize = 40 * 1024;
#else
constexpr int scratchBufSize = 0;
#endif
// An area of memory to use for input, output, and intermediate arrays.
constexpr int kTensorArenaSize = 375 * 1024 + scratchBufSize;
static uint8_t *tensor_arena;
//[kTensorArenaSize]; // Maybe we should move this to external
}  // namespace
tflite::MicroProfiler profiler; 
// Funciones para medir los ticks

// The name of this function is important for Arduino compatibility.
#include "esp_heap_caps.h"
#include "esp_log.h"
#define LED_PIN GPIO_NUM_4
#define I2C_SDA_PIN GPIO_NUM_14
#define I2C_SCL_PIN GPIO_NUM_15
#include "esp_heap_caps.h"
#include "esp_log.h"

void print_memory_statistics(const char* tag) {
  printf("%s\n", tag);
  printf("Total heap size: %d\n", heap_caps_get_total_size(MALLOC_CAP_8BIT));
  printf("Free heap size: %d\n", heap_caps_get_free_size(MALLOC_CAP_8BIT));
  printf("Total PSRAM size: %d\n", heap_caps_get_total_size(MALLOC_CAP_SPIRAM));
  printf("Free PSRAM size: %d\n", heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
}


// Function to initialize the TensorFlow model and resources.
void setup() {
    // Attempt to allocate memory in PSRAM first
    print_memory_statistics("Before allocation");
    tensor_arena = (uint8_t *) heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (tensor_arena == NULL) {
        printf("Couldn't allocate memory of %d bytes in PSRAM.\n", kTensorArenaSize);
        // Print memory statistics after failed PSRAM allocation attempt
        print_memory_statistics("After PSRAM allocation attempt");

        // Attempt to allocate memory in internal memory
        tensor_arena = (uint8_t *) heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
        if (tensor_arena == NULL) {
        printf("Couldn't allocate memory of %d bytes in internal memory either.\n", kTensorArenaSize);
        // Print memory statistics after failed internal memory allocation attempt
        print_memory_statistics("After internal memory allocation attempt");
        return;
        } else {
        printf("Successfully allocated %d bytes in internal memory.\n", kTensorArenaSize);
        // Print memory statistics after successful internal memory allocation
        print_memory_statistics("After successful internal memory allocation");
        }
    } else {
        printf("Successfully allocated %d bytes in PSRAM.\n", kTensorArenaSize);
        // Print memory statistics after successful PSRAM allocation
        print_memory_statistics("After successful PSRAM allocation");
    }

    // Map the model into a usable data structure. This doesn't involve any
    // copying or parsing, it's a very lightweight operation.
    model = tflite::GetModel(g_person_detect_model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        MicroPrintf("Model provided is schema version %d not equal to supported "
                    "version %d.", model->version(), TFLITE_SCHEMA_VERSION);
        return;
    }

    // Setup operation resolver.
    static tflite::MicroMutableOpResolver<7> micro_op_resolver;
    if (micro_op_resolver.AddQuantize() != kTfLiteOk) return;
    if (micro_op_resolver.AddFullyConnected() != kTfLiteOk) return;
    if (micro_op_resolver.AddMaxPool2D() != kTfLiteOk) return;
    if (micro_op_resolver.AddConv2D() != kTfLiteOk) return;
    if (micro_op_resolver.AddReshape() != kTfLiteOk) return;
    if (micro_op_resolver.AddSoftmax() != kTfLiteOk) return;
    if (micro_op_resolver.AddDequantize() != kTfLiteOk) return;

    // Build the interpreter.
    static tflite::MicroInterpreter static_interpreter(
        model, micro_op_resolver, tensor_arena, kTensorArenaSize);
    interpreter = &static_interpreter;

    // Allocate memory for the model's tensors and check for errors.
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        MicroPrintf("AllocateTensors() failed");
        return; // Error handling for tensor allocation failure.
    }

    // Get input tensor.
    input = interpreter->input(0);

#ifndef CLI_ONLY_INFERENCE
    // Initialize Camera and handle potential errors.
    if (InitCamera() != kTfLiteOk) {
        MicroPrintf("InitCamera failed\n");
        return; // Error handling for camera initialization failure.
    }
#endif
}

#ifndef CLI_ONLY_INFERENCE
// Main loop function for continuous image capture and inference.
void loop() {
    // Get image from provider and check for success.
    vTaskDelay(2);
    if (GetImage(kNumCols, kNumRows, kNumChannels, input->data.int8) != kTfLiteOk) {
        MicroPrintf("Image capture failed.");
        return; // Exit loop on image capture failure.
    }

    // Run the model on this input and ensure success.
    if (interpreter->Invoke() != kTfLiteOk) {
        MicroPrintf("Invoke failed.");
        return; // Exit loop on inference failure.
    }

    // Process inference results.
    TfLiteTensor* output = interpreter->output(0);
    // Extract scores for each class from output tensor.
    int8_t scores[7]; // Assuming 7 classes
    for (int i = 0; i < 7; ++i) {
        scores[i] = output->data.uint8[i]; // Access output tensor data
    }

    // Scale and convert scores to float for further processing.
    float scaled_scores[7];
    for (int i = 0; i < 7; ++i) {
        scaled_scores[i] = (scores[i] - output->params.zero_point) * output->params.scale;
    }

    // Respond to detection with scaled scores.
    RespondToDetection(scaled_scores[0], scaled_scores[1], scaled_scores[2], 
                       scaled_scores[3], scaled_scores[4], scaled_scores[5], 
                       scaled_scores[6]);
    vTaskDelay(500); // to avoid watchdog trigger
}
#endif

// Function to run inference with external image data.
void run_inference(void *ptr) {
    /* Convert from uint8 picture data to int8 */
    for (int i = 0; i < kNumCols * kNumRows; i++) {
        input->data.int8[i] = ((uint8_t *) ptr)[i] - 128; // Convert to signed int8
    }

    // Run the model on this input and ensure success.
    if (kTfLiteOk != interpreter->Invoke()) {
        MicroPrintf("Invoke failed.");
        return; // Error handling for inference failure.
    }

    TfLiteTensor* output = interpreter->output(0);
    // Extract scores for each class from output tensor.
    int8_t scores[7]; // Assuming 7 classes
    for (int i = 0; i < 7; ++i) {
        scores[i] = output->data.uint8[i]; // Access output tensor data
    }

    // Scale and convert scores to float for further processing.
    float scaled_scores[7];
    for (int i = 0; i < 7; ++i) {
        scaled_scores[i] = (scores[i] - output->params.zero_point) * output->params.scale;
    }

    // Respond to detection with scaled scores.
    RespondToDetection(scaled_scores[0], scaled_scores[1], scaled_scores[2], 
                       scaled_scores[3], scaled_scores[4], scaled_scores[5], 
                       scaled_scores[6]);
}