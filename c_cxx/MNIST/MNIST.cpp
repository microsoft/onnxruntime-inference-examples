// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#define UNICODE
#include <onnxruntime_cxx_api.h>
#include <windows.h>

// C++20 Standard Library
#include <array>
#include <cmath>
#include <format>
#include <memory>
#include <ranges>
#include <span>
#include <string>
#include <vector>

// Microsoft WIL for modern Windows programming
#include <wil/resource.h>
#include <wil/result.h>

#pragma comment(lib, "user32.lib")
#pragma comment(lib, "gdi32.lib")
#pragma comment(lib, "onnxruntime.lib")

// A C++20 concept to constrain the template to types that are random-access ranges of floats.
template <typename T>
concept FloatRange = std::ranges::random_access_range<T> && std::is_same_v<std::ranges::range_value_t<T>, float>;

// Applies the SoftMax function to a container of floats.
// Uses std::ranges::max for conciseness.
static void softmax(FloatRange auto& input) {
  const float rowmax = std::ranges::max(input);

  std::vector<float> y;
  y.reserve(std::ranges::size(input));
  float sum = 0.0f;

  for (const float value : input) {
    const float exp_val = std::exp(value - rowmax);
    y.push_back(exp_val);
    sum += exp_val;
  }

  for (size_t i = 0; i < std::ranges::size(input); ++i) {
    input[i] = y[i] / sum;
  }
}

// This is the structure to interface with the MNIST model
// After instantiation, set the input_image_ data to be the 28x28 pixel image of the number to recognize
// Then call Run() to fill in the results_ data with the probabilities of each
// result_ holds the index with highest probability (aka the number the model thinks is in the image)
struct MNIST {
  MNIST() {
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    input_tensor_ = Ort::Value::CreateTensor<float>(memory_info, input_image_.data(), input_image_.size(),
                                                    input_shape_.data(), input_shape_.size());
    output_tensor_ = Ort::Value::CreateTensor<float>(memory_info, results_.data(), results_.size(),
                                                     output_shape_.data(), output_shape_.size());
  }

  // Runs the inference and returns the digit with the highest probability.
  std::ptrdiff_t Run() {
    const char* input_names[] = {"Input3"};
    const char* output_names[] = {"Plus214_Output_0"};

    Ort::RunOptions run_options;
    session_.Run(run_options, input_names, &input_tensor_, 1, output_names, &output_tensor_, 1);
    softmax(results_);

    // Use std::ranges::max_element for a cleaner syntax.
    result_ = std::ranges::distance(results_.begin(), std::ranges::max_element(results_));
    return result_;
  }

  static constexpr const int width_ = 28;
  static constexpr const int height_ = 28;

  std::array<float, width_ * height_> input_image_{};
  std::array<float, 10> results_{};
  int64_t result_{0};

 private:
  Ort::Env env;
  Ort::Session session_{env, L"mnist.onnx", Ort::SessionOptions{nullptr}};

  Ort::Value input_tensor_{nullptr};
  std::array<int64_t, 4> input_shape_{1, 1, width_, height_};

  Ort::Value output_tensor_{nullptr};
  std::array<int64_t, 2> output_shape_{1, 10};
};

// --- Global Variables and Constants ---
const constexpr int drawing_area_inset_{4}; // Number of pixels to inset the top left of the drawing area
const constexpr int drawing_area_scale_{4}; // Number of times larger to make the drawing area compared to the shape inputs
const constexpr int drawing_area_width_{MNIST::width_ * drawing_area_scale_};
const constexpr int drawing_area_height_{MNIST::height_ * drawing_area_scale_};

std::unique_ptr<MNIST> g_mnist;
bool g_isPainting{};

// Use WIL's RAII wrappers for GDI objects to ensure they are always released.
wil::unique_hbitmap g_dib;
wil::unique_hdc g_hdcDib;
wil::unique_hbrush g_brushWinner{CreateSolidBrush(RGB(128, 255, 128))};
wil::unique_hbrush g_brushBars{CreateSolidBrush(RGB(128, 128, 255))};

// Helper struct to safely query DIBSECTION details.
struct DIBInfo : DIBSECTION {
  DIBInfo(HBITMAP hBitmap) {
    // Use WIL to throw an exception on failure, improving robustness.
    THROW_IF_WIN32_BOOL_FALSE(::GetObject(hBitmap, sizeof(DIBSECTION), this));
  }

  int Width() const noexcept { return dsBm.bmWidth; }
  int Height() const noexcept { return abs(dsBm.bmHeight); }
  void* Bits() const noexcept { return dsBm.bmBits; }
  int Pitch() const noexcept { return dsBm.bmWidthBytes; }
};

// Converts the 32bpp DIB into the model's required floating-point format.
// This version uses std::span for safe, bounded buffer manipulation.
void ConvertDibToMnist() {
  DIBInfo info(g_dib.get());

  // Create a span representing the source DIB pixel data.
  std::span<const std::byte> source_bits(static_cast<const std::byte*>(info.Bits()), info.Pitch() * info.Height());

  // Create a span over the destination float array.
  std::span<float> dest_floats = g_mnist->input_image_;
  std::ranges::fill(dest_floats, 0.0f);

  constexpr size_t bytes_per_pixel = 4;  // 32bpp

  for (int y = 0; y < MNIST::height_; ++y) {
    // Create a view into the current row of the source and destination.
    auto source_row = source_bits.subspan(y * info.Pitch(), MNIST::width_ * bytes_per_pixel);
    auto dest_row = dest_floats.subspan(y * MNIST::width_, MNIST::width_);

    for (int x = 0; x < MNIST::width_; ++x) {
      // Get the 32-bit pixel value (BGRA).
      uint32_t pixel = *reinterpret_cast<const uint32_t*>(source_row.data() + x * bytes_per_pixel);

      // The model expects a normalized float [0, 1].
      // We treat black pixels (0x000000) as the drawn digit (1.0f).
      dest_row[x] = (pixel == 0) ? 1.0f : 0.0f;
    }
  }
}

LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);

int APIENTRY wWinMain(_In_ HINSTANCE hInstance, _In_opt_ HINSTANCE, _In_ LPTSTR, _In_ int nCmdShow) {
  try {
    g_mnist = std::make_unique<MNIST>();

    WNDCLASSEX wc{};
    wc.cbSize = sizeof(WNDCLASSEX);
    wc.style = CS_HREDRAW | CS_VREDRAW;
    wc.lpfnWndProc = WndProc;
    wc.hInstance = hInstance;
    wc.hCursor = LoadCursor(NULL, IDC_ARROW);
    wc.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
    wc.lpszClassName = L"ONNXTest";
    THROW_LAST_ERROR_IF(RegisterClassEx(&wc) == 0);

    BITMAPINFO bmi{};
    bmi.bmiHeader.biSize = sizeof(bmi.bmiHeader);
    bmi.bmiHeader.biWidth = MNIST::width_;
    bmi.bmiHeader.biHeight = -MNIST::height_;  // Top-down DIB
    bmi.bmiHeader.biPlanes = 1;
    bmi.bmiHeader.biBitCount = 32;
    bmi.bmiHeader.biCompression = BI_RGB;

    void* bits = nullptr;
    g_dib.reset(CreateDIBSection(nullptr, &bmi, DIB_RGB_COLORS, &bits, nullptr, 0));
    THROW_IF_NULL_ALLOC(g_dib.get());

    g_hdcDib.reset(CreateCompatibleDC(nullptr));
    THROW_IF_NULL_ALLOC(g_hdcDib.get());

    auto dibSelection = wil::SelectObject(g_hdcDib.get(), g_dib.get());
    // This pen is created, selected, and automatically destroyed when penSelection goes out of scope.
    auto penSelection = wil::SelectObject(g_hdcDib.get(), CreatePen(PS_SOLID, 2, RGB(0, 0, 0)));

    RECT rect{0, 0, MNIST::width_, MNIST::height_};
    FillRect(g_hdcDib.get(), &rect, (HBRUSH)GetStockObject(WHITE_BRUSH));

    HWND hWnd = CreateWindow(L"ONNXTest", L"ONNX Runtime Sample - MNIST", WS_OVERLAPPEDWINDOW, CW_USEDEFAULT,
                             CW_USEDEFAULT, 512, 256, nullptr, nullptr, hInstance, nullptr);
    THROW_IF_WIN32_BOOL_FALSE(IsWindow(hWnd));

    ShowWindow(hWnd, nCmdShow);

    MSG msg;
    while (GetMessage(&msg, NULL, 0, 0)) {
      TranslateMessage(&msg);
      DispatchMessage(&msg);
    }
    return static_cast<int>(msg.wParam);
  }
  // Catch specific exception types for better error reporting.
  catch (const wil::ResultException& e) {
    MessageBoxA(nullptr, e.what(), "WIL Error", MB_OK | MB_ICONERROR);
  } catch (const Ort::Exception& e) {
    MessageBoxA(nullptr, e.what(), "ONNX Runtime Error", MB_OK | MB_ICONERROR);
  } catch (const std::exception& e) {
    MessageBoxA(nullptr, e.what(), "Standard C++ Error", MB_OK | MB_ICONERROR);
  }
  return 0;
  // All WIL unique_ handles are automatically cleaned up here.
}

LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam) {
  switch (message) {
    case WM_PAINT: {
      PAINTSTRUCT ps;
      HDC hdc = BeginPaint(hWnd, &ps);

      StretchBlt(hdc, drawing_area_inset_, drawing_area_inset_, drawing_area_width_, drawing_area_height_,
                 g_hdcDib.get(), 0, 0, MNIST::width_, MNIST::height_, SRCCOPY);

      {
        // This scope-based RAII guard ensures the pen and brush are restored after the Rectangle call.
        auto penSelection = wil::SelectObject(hdc, GetStockObject(BLACK_PEN));
        auto brushSelection = wil::SelectObject(hdc, GetStockObject(NULL_BRUSH));
        Rectangle(hdc, drawing_area_inset_, drawing_area_inset_, drawing_area_inset_ + drawing_area_width_,
                  drawing_area_inset_ + drawing_area_height_);
      }

      constexpr int graphs_left = drawing_area_inset_ + drawing_area_width_ + 5;
      constexpr int graph_width = 64;

      auto [least, greatest] = std::ranges::minmax(g_mnist->results_);
      auto range = greatest - least;
      if (range == 0.0f) range = 1.0f;  // Avoid division by zero.

      int graphs_zero = static_cast<int>(graphs_left - least * graph_width / range);

      RECT rcWinner{graphs_left, static_cast<LONG>(g_mnist->result_) * 16, graphs_left + graph_width + 128,
                    static_cast<LONG>(g_mnist->result_ + 1) * 16};
      FillRect(hdc, &rcWinner, g_brushWinner.get());

      SetBkMode(hdc, TRANSPARENT);
      {
        // Create a new RAII guard for the bar brush. It will be restored when the scope ends.
        auto barBrushSelection = wil::SelectObject(hdc, g_brushBars.get());
        for (unsigned i = 0; i < 10; ++i) {
          int y = 16 * i;
          float result = g_mnist->results_[i];
          auto value = std::format(L"{:2}: {:.2f}", i, result);
          TextOutW(hdc, graphs_left + graph_width + 5, y, value.data(), static_cast<int>(value.length()));
          Rectangle(hdc, graphs_zero, y + 1, static_cast<int>(graphs_zero + result * graph_width / range), y + 14);
        }
      }

      MoveToEx(hdc, graphs_zero, 0, nullptr);
      LineTo(hdc, graphs_zero, 16 * 10);

      EndPaint(hWnd, &ps);
      return 0;
    }

    case WM_LBUTTONDOWN: {
      SetCapture(hWnd);
      g_isPainting = true;
      // FIX: Replaced GET_X_LPARAM and GET_Y_LPARAM macros
      int x = ((int)(short)LOWORD(lParam) - drawing_area_inset_) / drawing_area_scale_;
      int y = ((int)(short)HIWORD(lParam) - drawing_area_inset_) / drawing_area_scale_;
      MoveToEx(g_hdcDib.get(), x, y, nullptr);
      return 0;
    }

    case WM_MOUSEMOVE:
      if (g_isPainting) {
        // FIX: Replaced GET_X_LPARAM and GET_Y_LPARAM macros
        int x = ((int)(short)LOWORD(lParam) - drawing_area_inset_) / drawing_area_scale_;
        int y = ((int)(short)HIWORD(lParam) - drawing_area_inset_) / drawing_area_scale_;
        LineTo(g_hdcDib.get(), x, y);
        InvalidateRect(hWnd, nullptr, false);
      }
      return 0;

    case WM_CAPTURECHANGED:
      g_isPainting = false;
      return 0;

    case WM_LBUTTONUP:
      ReleaseCapture();
      ConvertDibToMnist();
      g_mnist->Run();
      InvalidateRect(hWnd, nullptr, true);
      return 0;

    case WM_RBUTTONDOWN: {
      RECT rect{0, 0, MNIST::width_, MNIST::height_};
      FillRect(g_hdcDib.get(), &rect, (HBRUSH)GetStockObject(WHITE_BRUSH));
      InvalidateRect(hWnd, nullptr, false);
      return 0;
    }

    case WM_DESTROY:
      PostQuitMessage(0);
      return 0;
  }
  return DefWindowProc(hWnd, message, wParam, lParam);
}
