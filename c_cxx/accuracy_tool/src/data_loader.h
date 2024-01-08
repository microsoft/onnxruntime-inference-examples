// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include <filesystem>
#include <memory>
#include <vector>

#include "basic_utils.h"
#include "model_io_utils.h"

/// <summary>
/// Load raw input or output data for a given set of dataset paths. For example, this can be used to load all
/// input_XXX.raw files for a particular model.
/// </summary>
/// <param name="dataset_paths">The directories containing raw data files</param>
/// <param name="io_infos">Type and shape information for the inputs or outputs of a model</param>
/// <param name="data_file_prefix">The prefix for the data file names (e.g., "input_" or "output_")</param>
/// <param name="dataset_data">Output buffer into which to store loaded data</param>
/// <returns>True on success</returns>
bool LoadIODataFromDisk(const std::vector<std::filesystem::path>& dataset_paths, const std::vector<IOInfo>& io_infos,
                        const char* data_file_prefix, std::vector<std::unique_ptr<char[]>>& dataset_data);
