# Changelog

## [0.8.2](https://github.com/huanglizhuo/QwenASR/compare/qwen-asr-cli-v0.8.1...qwen-asr-cli-v0.8.2) (2026-07-12)


### Performance Improvements

* pool-parallel GEMM slices and E-core-aware default threads ([9a0c19e](https://github.com/huanglizhuo/QwenASR/commit/9a0c19ea26c94a8989e77ecf7f32219055911335))


### Dependencies

* The following workspace dependencies were updated
  * dependencies
    * qwen-asr bumped from 0.8.1 to 0.8.2

## [0.8.1](https://github.com/huanglizhuo/QwenASR/compare/qwen-asr-cli-v0.8.0...qwen-asr-cli-v0.8.1) (2026-07-09)


### Dependencies

* The following workspace dependencies were updated
  * dependencies
    * qwen-asr bumped from 0.8.0 to 0.8.1

## [0.8.0](https://github.com/huanglizhuo/QwenASR/compare/qwen-asr-cli-v0.7.4...qwen-asr-cli-v0.8.0) (2026-07-08)


### Features

* add structured subtitle outputs ([14c39c0](https://github.com/huanglizhuo/QwenASR/commit/14c39c05103be978b8d0e374520b9965d8eaf6ee))


### Bug Fixes

* restore default decode preamble and correct cue-grouping rules ([757647c](https://github.com/huanglizhuo/QwenASR/commit/757647c211172df4a063a8af9abbdc8b792c4b31))


### Dependencies

* The following workspace dependencies were updated
  * dependencies
    * qwen-asr bumped from 0.7.5 to 0.8.0

## [0.7.4](https://github.com/huanglizhuo/QwenASR/compare/qwen-asr-cli-v0.7.3...qwen-asr-cli-v0.7.4) (2026-07-03)


### Dependencies

* The following workspace dependencies were updated
  * dependencies
    * qwen-asr bumped from 0.7.4 to 0.7.5

## [0.7.3](https://github.com/huanglizhuo/QwenASR/compare/qwen-asr-cli-v0.7.2...qwen-asr-cli-v0.7.3) (2026-07-02)


### Dependencies

* The following workspace dependencies were updated
  * dependencies
    * qwen-asr bumped from 0.7.3 to 0.7.4

## [0.7.2](https://github.com/huanglizhuo/QwenASR/compare/qwen-asr-cli-v0.7.1...qwen-asr-cli-v0.7.2) (2026-06-27)


### Bug Fixes

* auto-sync qwen-asr version into qwen-asr-cli via cargo-workspace ([#36](https://github.com/huanglizhuo/QwenASR/issues/36)) ([e34ba23](https://github.com/huanglizhuo/QwenASR/commit/e34ba236b1ef5d9b1ba9d3998ebeb3ce8d50ebaa))


### Dependencies

* The following workspace dependencies were updated
  * dependencies
    * qwen-asr bumped from 0.7.2 to 0.7.3

## [0.7.1](https://github.com/huanglizhuo/QwenASR/compare/qwen-asr-cli-v0.7.0...qwen-asr-cli-v0.7.1) (2026-06-13)


### Performance Improvements

* add startup-phase profile breakdown (A6) ([8eba33f](https://github.com/huanglizhuo/QwenASR/commit/8eba33fb91003fcf65ef7474df412b5fd1ec66eb))
* default thread count to performance cores (post-E8) ([44d6bad](https://github.com/huanglizhuo/QwenASR/commit/44d6bade14ca26fa2fa1daf2928cc0d579b45dc6))
* overlap audio front-end loading with model load (A2) ([b219874](https://github.com/huanglizhuo/QwenASR/commit/b2198749cbcab80500e2a04d47a22679acbaac37))

## [0.7.0](https://github.com/huanglizhuo/QwenASR/compare/qwen-asr-cli-v0.6.0...qwen-asr-cli-v0.7.0) (2026-05-10)


### Features

* document benchmarked ASR performance ([f3b33a0](https://github.com/huanglizhuo/QwenASR/commit/f3b33a0863c815fedbdac18650e2b673af53efbe))

## [0.6.0](https://github.com/huanglizhuo/QwenASR/compare/qwen-asr-cli-v0.5.0...qwen-asr-cli-v0.6.0) (2026-03-21)


### Features

* add missing parameter to qwen asr offline model ([f56e8b1](https://github.com/huanglizhuo/QwenASR/commit/f56e8b1e58731344fad92a7ed38c59a9f09267f6))
* add missing parameter to qwen asr offline model ([6d1e38d](https://github.com/huanglizhuo/QwenASR/commit/6d1e38da19cbae46c2afe2e1af03a5d437679ef8))
* improve the live stream performace on macos ([ba47230](https://github.com/huanglizhuo/QwenASR/commit/ba47230403f897bde2486b3235cfd3f5ca24e293))
* publish qwen-asr-cli binary to crates.io ([31ae992](https://github.com/huanglizhuo/QwenASR/commit/31ae99221d71fdcd6c40b6cfe4d77e7f67642b76))
* support live from blackhold for macos for qwen-asr-cli ([724ead1](https://github.com/huanglizhuo/QwenASR/commit/724ead1fe121d0ed0a0f7ef874142f665e7d0da3))


### Bug Fixes

* bump qwen-asr dependency version for cli ([efec6b1](https://github.com/huanglizhuo/QwenASR/commit/efec6b14dfb07934c7e6cbb459d26573dcfe5912))
* clean up the code ([346d112](https://github.com/huanglizhuo/QwenASR/commit/346d112595c0d93f58c54339a7f32ca6e1e648d8))
* publish 0.2.3 with tag-driven flow ([3637ec8](https://github.com/huanglizhuo/QwenASR/commit/3637ec80f5519ecbd0a034f6c1f23f78156cd0fe))
* publish 0.2.3 with tag-driven flow ([e7bbd18](https://github.com/huanglizhuo/QwenASR/commit/e7bbd18dc009c3bd87f32e2346c196f65c618b19))
* **qwen-asr-cli:** add homepage metadata to trigger release ([642c6a2](https://github.com/huanglizhuo/QwenASR/commit/642c6a2dbdcd01c4351d6a56dfdf2c6b99fa5488))
* **qwen-asr-cli:** update qwen-asr dependency to v0.4.2 ([d9915ea](https://github.com/huanglizhuo/QwenASR/commit/d9915ea8eba41f3a7a129d0ea5c6cea939a33c6d))
* **qwen-asr-cli:** use workspace dependency to keep qwen-asr version in sync ([5ac7e01](https://github.com/huanglizhuo/QwenASR/commit/5ac7e01bdaf95b3026eec7a12f04e9f5d78d0f3b))
* release flow for cli ([8dfc4b7](https://github.com/huanglizhuo/QwenASR/commit/8dfc4b7820ceef98efd992974ab3a0b7fcdd9b10))
* trigger patch release 0.2.1 for flutter ([b5785f9](https://github.com/huanglizhuo/QwenASR/commit/b5785f9e0a6e4cab3a4796bbd1bd401876ea5926))
* update the release flow to support PAT ([2b9be6c](https://github.com/huanglizhuo/QwenASR/commit/2b9be6c21b7e74e51bf1d1f15e6959679db70542))

## [0.4.0](https://github.com/huanglizhuo/QwenASR/compare/qwen-asr-cli-v0.3.4...qwen-asr-cli-v0.4.0) (2026-03-19)


### Features

* add missing parameter to qwen asr offline model ([f56e8b1](https://github.com/huanglizhuo/QwenASR/commit/f56e8b1e58731344fad92a7ed38c59a9f09267f6))
* add missing parameter to qwen asr offline model ([6d1e38d](https://github.com/huanglizhuo/QwenASR/commit/6d1e38da19cbae46c2afe2e1af03a5d437679ef8))
* improve the live stream performace on macos ([ba47230](https://github.com/huanglizhuo/QwenASR/commit/ba47230403f897bde2486b3235cfd3f5ca24e293))
* publish qwen-asr-cli binary to crates.io ([31ae992](https://github.com/huanglizhuo/QwenASR/commit/31ae99221d71fdcd6c40b6cfe4d77e7f67642b76))
* support live from blackhold for macos for qwen-asr-cli ([724ead1](https://github.com/huanglizhuo/QwenASR/commit/724ead1fe121d0ed0a0f7ef874142f665e7d0da3))


### Bug Fixes

* bump qwen-asr dependency version for cli ([efec6b1](https://github.com/huanglizhuo/QwenASR/commit/efec6b14dfb07934c7e6cbb459d26573dcfe5912))
* clean up the code ([346d112](https://github.com/huanglizhuo/QwenASR/commit/346d112595c0d93f58c54339a7f32ca6e1e648d8))
* publish 0.2.3 with tag-driven flow ([3637ec8](https://github.com/huanglizhuo/QwenASR/commit/3637ec80f5519ecbd0a034f6c1f23f78156cd0fe))
* publish 0.2.3 with tag-driven flow ([e7bbd18](https://github.com/huanglizhuo/QwenASR/commit/e7bbd18dc009c3bd87f32e2346c196f65c618b19))
* **qwen-asr-cli:** add homepage metadata to trigger release ([642c6a2](https://github.com/huanglizhuo/QwenASR/commit/642c6a2dbdcd01c4351d6a56dfdf2c6b99fa5488))
* **qwen-asr-cli:** update qwen-asr dependency to v0.4.2 ([d9915ea](https://github.com/huanglizhuo/QwenASR/commit/d9915ea8eba41f3a7a129d0ea5c6cea939a33c6d))
* release flow for cli ([8dfc4b7](https://github.com/huanglizhuo/QwenASR/commit/8dfc4b7820ceef98efd992974ab3a0b7fcdd9b10))
* trigger patch release 0.2.1 for flutter ([b5785f9](https://github.com/huanglizhuo/QwenASR/commit/b5785f9e0a6e4cab3a4796bbd1bd401876ea5926))
* update the release flow to support PAT ([2b9be6c](https://github.com/huanglizhuo/QwenASR/commit/2b9be6c21b7e74e51bf1d1f15e6959679db70542))

## [0.3.4](https://github.com/huanglizhuo/QwenASR/compare/qwen-asr-cli-v0.3.3...qwen-asr-cli-v0.3.4) (2026-03-19)


### Bug Fixes

* **qwen-asr-cli:** update qwen-asr dependency to v0.4.2 ([d9915ea](https://github.com/huanglizhuo/QwenASR/commit/d9915ea8eba41f3a7a129d0ea5c6cea939a33c6d))

## [0.3.3](https://github.com/huanglizhuo/QwenASR/compare/qwen-asr-cli-v0.3.2...qwen-asr-cli-v0.3.3) (2026-03-14)


### Bug Fixes

* clean up the code ([346d112](https://github.com/huanglizhuo/QwenASR/commit/346d112595c0d93f58c54339a7f32ca6e1e648d8))

## [0.3.2](https://github.com/huanglizhuo/QwenASR/compare/qwen-asr-cli-v0.3.1...qwen-asr-cli-v0.3.2) (2026-03-13)


### Bug Fixes

* release flow for cli ([8dfc4b7](https://github.com/huanglizhuo/QwenASR/commit/8dfc4b7820ceef98efd992974ab3a0b7fcdd9b10))

## [0.3.1](https://github.com/huanglizhuo/QwenASR/compare/qwen-asr-cli-v0.3.0...qwen-asr-cli-v0.3.1) (2026-02-23)


### Bug Fixes

* bump qwen-asr dependency version for cli ([efec6b1](https://github.com/huanglizhuo/QwenASR/commit/efec6b14dfb07934c7e6cbb459d26573dcfe5912))

## [0.3.0](https://github.com/huanglizhuo/QwenASR/compare/qwen-asr-cli-v0.2.4...qwen-asr-cli-v0.3.0) (2026-02-23)


### Features

* add missing parameter to qwen asr offline model ([f56e8b1](https://github.com/huanglizhuo/QwenASR/commit/f56e8b1e58731344fad92a7ed38c59a9f09267f6))
* add missing parameter to qwen asr offline model ([6d1e38d](https://github.com/huanglizhuo/QwenASR/commit/6d1e38da19cbae46c2afe2e1af03a5d437679ef8))
* improve the live stream performace on macos ([ba47230](https://github.com/huanglizhuo/QwenASR/commit/ba47230403f897bde2486b3235cfd3f5ca24e293))
* publish qwen-asr-cli binary to crates.io ([31ae992](https://github.com/huanglizhuo/QwenASR/commit/31ae99221d71fdcd6c40b6cfe4d77e7f67642b76))
* support live from blackhold for macos for qwen-asr-cli ([724ead1](https://github.com/huanglizhuo/QwenASR/commit/724ead1fe121d0ed0a0f7ef874142f665e7d0da3))


### Bug Fixes

* publish 0.2.3 with tag-driven flow ([3637ec8](https://github.com/huanglizhuo/QwenASR/commit/3637ec80f5519ecbd0a034f6c1f23f78156cd0fe))
* publish 0.2.3 with tag-driven flow ([e7bbd18](https://github.com/huanglizhuo/QwenASR/commit/e7bbd18dc009c3bd87f32e2346c196f65c618b19))
* trigger patch release 0.2.1 for flutter ([b5785f9](https://github.com/huanglizhuo/QwenASR/commit/b5785f9e0a6e4cab3a4796bbd1bd401876ea5926))
* update the release flow to support PAT ([2b9be6c](https://github.com/huanglizhuo/QwenASR/commit/2b9be6c21b7e74e51bf1d1f15e6959679db70542))
