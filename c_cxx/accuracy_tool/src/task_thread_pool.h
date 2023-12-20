// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>

#include "acc_task.h"
#include "basic_utils.h"

/// <summary>
/// A class that runs a fixed set of inference tasks using a pool of N threads.
/// Usage example:
///     TaskThreadPool pool(2);     // Pool with N (2) threads.
///     Span<Task> tasks = /*...*/; // Fixed set of M tasks.
///
///     pool.CompleteTasks(tasks);  // N + 1 threads (2 + 1) will complete the tasks.
///                                 // The main thread helps too (blocking)!
/// </summary>
class TaskThreadPool {
 public:
  TaskThreadPool(size_t num_threads);
  ~TaskThreadPool();

  /// <summary>
  /// Blocks the calling thread until all provided tasks are completed. The calling thread
  /// also helps complete the tasks.
  /// </summary>
  /// <param name="tasks">The fixed set of tasks to complete.</param>
  void CompleteTasks(Span<Task> tasks);

 private:
  void ThreadEntry();
  bool RunNextTask();

  std::mutex lock_;
  std::condition_variable signal_;
  bool shutdown_ = false;
  Span<Task> tasks_;
  std::atomic<size_t> next_task_index_ = 0;
  std::atomic<size_t> tasks_completed_ = 0;
  std::vector<std::thread> threads_;
};
