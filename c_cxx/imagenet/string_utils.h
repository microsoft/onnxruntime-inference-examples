#pragma once
#include <string>
#include <string_view>

std::string ToMBString(std::wstring_view s);
std::string ToUTF8String(std::wstring_view s);