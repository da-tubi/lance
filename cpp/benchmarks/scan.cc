//  Copyright 2022 Lance Authors
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#include <arrow/array.h>
#include <arrow/filesystem/api.h>
#include <arrow/record_batch.h>
#include <arrow/util/string.h>
#include <fmt/chrono.h>
#include <fmt/format.h>
#include <parquet/arrow/reader.h>

#include <argparse/argparse.hpp>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <string>

#include "bench_utils.h"
#include "lance/format/schema.h"
#include "lance/io/reader.h"
#include "lance/io/scanner.h"

std::vector<std::string> GetPetNames(const std::string& uri) {
  std::vector<std::string> names;
  auto file = OpenUri(uri);
  auto size = file->GetSize().ValueOrDie();
  auto buf = file->ReadAt(0, size).ValueOrDie();
  auto content = buf->ToString();

  auto lines = ::arrow::internal::SplitString(content, '\n');
  for (std::size_t i = 6; i < lines.size(); i++) {
    auto line = lines[i].to_string();
    auto comps = ::arrow::internal::SplitString(line, ' ');
    if (!comps[0].empty()) {
      names.emplace_back(comps[0].to_string());
    }
  }
  return names;
}

void run_scan(const std::string& uri, const std::vector<std::string>& columns, int batch_size) {
  auto scanner = OpenScanner(uri, columns, std::nullopt, batch_size);
  auto batches = scanner->ScanBatches().ValueOrDie();

  fmt::print("Start to run\n");
  auto start = std::chrono::steady_clock::now();
  int total = 0;
  while (true) {
    auto n = batches.Next().ValueOrDie();
    if (!n.record_batch) {
      break;
    }
    total++;
  }
  auto end = std::chrono::steady_clock::now();

  fmt::print("{} batches, bath_size={} mean time: {} \n",
             total,
             batch_size,
             std::chrono::duration_cast<std::chrono::microseconds>((end - start) / total));
}

void scan_pet_xml(int batch_size) {
  auto pet_names = GetPetNames("s3://eto-public/datasets/oxford_pet/annotations/list.txt");

  std::vector<std::chrono::duration<double>> batch_times;
  auto start = std::chrono::steady_clock::now();
  std::size_t total = 0;
  for (std::size_t start_pos = 0, last_total = 0; start_pos < pet_names.size(); total++) {
    // When use batch_size larger than 8, raises
    // IOError: When resolving region for bucket 'eto-public':
    // AWS Error [code 99]: curlCode: 28, Timeout was reached
    for (int c = 0; c < batch_size / 8; c++) {
      auto len = std::min(static_cast<size_t>(8), pet_names.size() - start_pos);
      fmt::print("read at: {}, batch={}\n", start_pos, batch_size);

      std::vector<std::future<void>> futures;
      for (std::size_t i = 0; i < len; i++) {
        futures.emplace_back(std::async(
            std::launch::async,
            [&](auto start, auto idx) {
              auto name = pet_names[start + idx];
              ReadAll(fmt::format("s3://eto-public/datasets/oxford_pet/images/{}.jpg", name), true);
              ReadAll(
                  fmt::format("s3://eto-public/datasets/oxford_pet/annotations/xmls/{}.xml", name),
                  true);
            },
            start_pos,
            i));
      };
      for (auto& f : futures) {
        f.wait();
      }
      start_pos += len;
    }

    if (start_pos - last_total == 0) {
      fmt::print("Progress: ={}\n", start_pos);
      last_total = start_pos;
    }
  }
  auto end = std::chrono::steady_clock::now();
  fmt::print("Total time: {}\n",
             std::chrono::duration_cast<std::chrono::microseconds>((end - start) / total));
}

void scan_coco_json(const std::string& uri,
                    const std::vector<std::string>& columns,
                    int batch_size) {
  auto scanner = OpenScanner(uri, columns, std::nullopt, batch_size);
  auto batches = scanner->ScanBatches().ValueOrDie();
  std::vector<std::chrono::duration<double>> batch_times;
  int nreads = 0;
  while (nreads < 1000) {
    auto start = std::chrono::steady_clock::now();
    auto n = batches.Next().ValueOrDie();
    if (!n.record_batch) {
      break;
    }
    nreads += n.record_batch->num_rows();
    fmt::print("Reading batch: {}\n", nreads);
    auto images =
        std::static_pointer_cast<arrow::StructArray>(n.record_batch->GetColumnByName("image"));
    auto uris = std::static_pointer_cast<arrow::StringArray>(images->GetFieldByName("uri"));

    std::vector<std::string> uri_vector;
    for (int64_t i = 0; i < uris->length(); i++) {
      uri_vector.emplace_back(uris->Value(i));
    }
    ReadAll(uri_vector, true);
    auto end = std::chrono::steady_clock::now();
    batch_times.emplace_back(end - start);
  }
  fmt::print("Mean time: {}\n",
             std::reduce(batch_times.begin(), batch_times.end()).count() / batch_times.size());
}

int main(int argc, char** argv) {
  argparse::ArgumentParser parser("scan");

  parser.add_argument("-b", "--batch_size")
      .help("Set batch size")
      .default_value(32)
      .scan<'i', int>();
  parser.add_argument("-c", "--column")
      .append()
      .help("Columns to read")
      .default_value<std::vector<std::string>>({});
  parser.add_argument("file").help("input file");

  try {
    parser.parse_args(argc, argv);
  } catch (const std::runtime_error& err) {
    std::cerr << err.what() << std::endl;
    std::cerr << parser;
    std::exit(1);
  }

  auto batch_size = parser.get<int>("batch_size");
  auto cols = parser.get<std::vector<std::string>>("column");
  auto input = parser.get("file");

  run_scan(input, cols, batch_size);
  //  scan_pet_xml(batch_size);
  //  scan_coco_json(batch_size);

  return 0;
}