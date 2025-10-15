#include <string>
#include <vector>

namespace test {
namespace trt_ep {

std::string ToUTF8String(std::wstring_view s);
std::wstring ToWideString(std::string_view s);

#define ENFORCE(condition, ...)                          \
  do {                                                   \
    if (!(condition)) {                                  \
      throw std::runtime_error(std::string(__VA_ARGS__)); \
    }                                                    \
  } while (false)

#define THROW(...) throw std::runtime_error(std::string(__VA_ARGS__));

#define RETURN_IF_ORTSTATUS_ERROR(fn) RETURN_IF_ERROR(fn)

#define RETURN_IF_ERROR(fn)    \
  do {                         \
    OrtStatus* _status = (fn); \
    if (_status != nullptr) {  \
      return _status;          \
    }                          \
  } while (0)

#define RETURN_IF_ORT_STATUS_ERROR(fn) \
  do {                                 \
    auto _status = (fn);               \
    if (!_status.IsOK()) {             \
      return _status;                  \
    }                                  \
  } while (0)

#define RETURN_IF(cond, ...)                                                           \
  do {                                                                                 \
    if ((cond)) {                                                                      \
      return Ort::GetApi().CreateStatus(ORT_EP_FAIL, std::string(__VA_ARGS__).c_str()); \
    }                                                                                  \
  } while (0)

#define RETURN_IF_NOT(condition, ...) RETURN_IF(!(condition), __VA_ARGS__)

#define MAKE_STATUS(error_code, msg) Ort::GetApi().CreateStatus(error_code, (msg));

#define THROW_IF_ERROR(expr)                         \
  do {                                               \
    auto _status = (expr);                           \
    if (_status != nullptr) {                        \
      std::ostringstream oss;                        \
      oss << Ort::GetApi().GetErrorMessage(_status); \
      Ort::GetApi().ReleaseStatus(_status);          \
      throw std::runtime_error(oss.str());           \
    }                                                \
  } while (0)

#define RETURN_FALSE_AND_PRINT_IF_ERROR(fn)                            \
  do {                                                                 \
    OrtStatus* status = (fn);                                          \
    if (status != nullptr) {                                           \
      std::cerr << Ort::GetApi().GetErrorMessage(status) << std::endl; \
      return false;                                                    \
    }                                                                  \
  } while (0)

void CreateBaseModel(const std::string& model_path,
                     const std::string& graph_name,
                     const std::vector<int64_t>& dims,
                     bool add_non_zero_node = false);
}
}