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

#include <arrow/filesystem/api.h>
#include <arrow/io/api.h>
#include <arrow/result.h>
#include <arrow/scalar.h>
#include <arrow/util/string.h>
#include <arrow/util/uri.h>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <parquet/arrow/reader.h>

#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <filesystem>
#include <future>
#include <random>
#include <string>

#include "bench_utils.h"
#include "lance/arrow/reader.h"

namespace fs = std::filesystem;
std::string input_file;

TEST_CASE("S3 Random Access Baseline") {
  fmt::print("Running against {}\n", input_file);
  auto fs = arrow::fs::FileSystemFromUri(input_file).ValueOrDie();
  auto f = fs->OpenInputFile(input_file.substr(std::string("s3://").size())).ValueOrDie();

  auto file_size = f->GetSize().ValueOrDie();

  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_int_distribution<int64_t> dist(0, file_size);

  auto read = [&](auto nbytes) {
    auto roff = dist(mt);
    auto offset = std::min(roff, file_size - nbytes);
    return f->ReadAt(offset, nbytes);
  };

  auto async_read = [&](auto nbytes) {
    std::vector<std::future<::arrow::Result<std::shared_ptr<arrow::Buffer>>>> futures;
    for (int i = 0; i < 20; i++) {
      futures.emplace_back(std::async(std::launch::async, read, nbytes));
    }
    std::for_each(futures.begin(), futures.end(), [](auto&& f) { f.wait(); });
    return 1;
  };

  BENCHMARK(fmt::format("256B")) { return read(256); };

  BENCHMARK("1k") { return read(1024); };

  BENCHMARK("8k") { return read(1024 * 8); };

  BENCHMARK("64k") { return read(1024 * 64); };

  BENCHMARK("256k") { return read(1024 * 256); };

  BENCHMARK("1M") { return read(1024 * 1024); };

  BENCHMARK("4M") { return read(1024 * 1024 * 4); };

  BENCHMARK("8M") { return read(1024 * 1024 * 8); };

  BENCHMARK("16M") { return read(1024 * 1024 * 16); };

  BENCHMARK("Async 20@256B") { return async_read(256); };

  BENCHMARK("Async 20 @ 1K") { return async_read(1024); };

  BENCHMARK("Async 20@ 8K") { return async_read(1024 * 8); };

  BENCHMARK("Async 20@ 64K") { return async_read(1024 * 64); };
}

TEST_CASE("Parquet") {
  INFO("Parquet point query: " << input_file);
  auto f = OpenUri(input_file);
  std::unique_ptr<parquet::arrow::FileReader> reader;
  auto status = parquet::arrow::OpenFile(f, ::arrow::default_memory_pool(), &reader);
  INFO("Open parquet file: " << status.message());
  CHECK(status.ok());
  fmt::print("Open parquet file: {}\n", input_file);
  fmt::print("Number of row groups: {}\n", reader->num_row_groups());

  auto num_row_groups = reader->num_row_groups();
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_int_distribution<int32_t> dist(0, num_row_groups - 1);

  auto read = [&]() {
    auto idx = dist(mt);
    auto row_group = reader->RowGroup(idx);
    std::shared_ptr<::arrow::Table> tbl;
    CHECK(row_group->ReadTable(&tbl).ok());
    auto num_rows = tbl->num_rows();
    std::uniform_int_distribution<int64_t> row_dist(0, num_rows);
    auto row = tbl->Slice(row_dist(mt), 1);
  };

  BENCHMARK("Single Thread") { return read(); };
}

TEST_CASE("Lance") {
  auto f = OpenUri(input_file);
  auto reader = ::lance::arrow::FileReader::Make(f).ValueOrDie();
  auto length = reader->length();

  fmt::print("Open parquet file: {}", input_file);
  fmt::print("Number of CHUNKS: {}, rows={}", reader->num_chunks(), length);

  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_int_distribution<int32_t> dist(1, length - 1);

  auto read = [&]() {
    auto idx = dist(mt);
    auto row = reader->Get(idx);
    CHECK(row.ok());
  };

  BENCHMARK("Single Thread") { return read(); };
}

TEST_CASE("Parquet with external images") {
  auto f = OpenUri(input_file);
  std::unique_ptr<parquet::arrow::FileReader> reader;
  auto status = parquet::arrow::OpenFile(f, ::arrow::default_memory_pool(), &reader);

  auto num_row_groups = reader->num_row_groups();
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_int_distribution<int32_t> dist(0, num_row_groups - 1);

  BENCHMARK("Single Thread") {
    auto idx = dist(mt);
    auto row_group = reader->RowGroup(idx);
    std::shared_ptr<::arrow::Table> tbl;
    CHECK(row_group->ReadTable(&tbl).ok());
    auto num_rows = tbl->num_rows();
    std::uniform_int_distribution<int64_t> row_dist(0, num_rows);
    auto row = tbl->Slice(row_dist(mt), 1);
    auto img_arr = row->GetColumnByName("image")->View(::arrow::utf8()).ValueOrDie();

    std::uniform_int_distribution<int32_t> img_dist(0, img_arr->length() - 1);
    auto img_idx = img_dist(mt);
    auto img_uri = img_arr->GetScalar(img_idx).ValueOrDie();
    // fmt::print("ROW IS: {}\n", img_uri->ToString());
    ReadAll(img_uri->ToString());
  };
}

TEST_CASE("pet.xml") {
  auto l = OpenUri("pet_list.txt");
  auto size = l->GetSize().ValueOrDie();
  auto buf = l->ReadAt(0, size).ValueOrDie();
  auto content = buf->ToString();

  auto lines = ::arrow::internal::SplitString(content, '\n');
  auto total = lines.size() - 1;

  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_int_distribution<int32_t> dist(0, total - 1);

  auto read = [&]() {
    auto idx = dist(mt);
    auto line = lines[idx].to_string();
    auto comps = ::arrow::internal::SplitString(line, ' ');

    ReadAll(fmt::format("s3://eto-public/datasets/oxford_pet/annotations/xmls/{}.xml",
                        comps[0].to_string()),
            true);
    ReadAll(fmt::format("s3://eto-public/datasets/oxford_pet/images/{}.jpg", comps[0].to_string()),
            true);
  };

  BENCHMARK("S3 Read") { return read(); };

  auto local_read = [&]() {
    auto idx = dist(mt);
    auto line = lines[idx].to_string();
    auto comps = ::arrow::internal::SplitString(line, ' ');

    ReadAll(
        fmt::format("/home/ubuntu/data/oxford_pet/annotations/xmls/{}.xml", comps[0].to_string()),
        true);
    ReadAll(fmt::format("/home/ubuntu/data/oxford_pet/images/{}.jpg", comps[0].to_string()), true);
  };

  BENCHMARK("Local Read") { return local_read(); };
}

int main(int argc, char* argv[]) {
  Catch::Session session;
  using namespace Catch::Clara;
  auto cli = session.cli() | Opt(input_file, "input file")["--uri"]("input file for benchmark");

  session.cli(cli);

  int returnCode = session.applyCommandLine(argc, argv);
  if (returnCode != 0) return returnCode;

  return session.run();
}