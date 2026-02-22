Pod::Spec.new do |s|
  s.name             = 'qwen_asr'
  s.version          = '0.1.0'
  s.summary          = 'CPU-only Qwen3-ASR inference for Flutter.'
  s.description      = 'Flutter plugin wrapping qwen-asr, a pure-Rust CPU-only Qwen3-ASR engine.'
  s.homepage         = 'https://github.com/user/qwen-asr'
  s.license          = { :file => '../LICENSE' }
  s.author           = { 'qwen-asr' => 'noreply@example.com' }
  s.source           = { :path => '.' }
  s.source_files = 'Classes/**/*'
  s.dependency 'FlutterMacOS'
  s.platform = :osx, '10.15'
  s.frameworks = 'Accelerate'

  s.script_phase = {
    :name => 'Build Rust library',
    :script => 'sh "$PODS_TARGET_SRCROOT/../cargokit/build_pod.sh" ../rust rust_lib_qwen_asr',
    :execution_position => :before_compile,
    :input_files => ['${BUILT_PRODUCTS_DIR}/cargokit_phony'],
    :output_files => ["${BUILT_PRODUCTS_DIR}/librust_lib_qwen_asr.a"],
  }
  s.pod_target_xcconfig = {
    'DEFINES_MODULE' => 'YES',
    'EXCLUDED_ARCHS[sdk=iphonesimulator*]' => 'i386',
    'OTHER_LDFLAGS' => '-force_load ${BUILT_PRODUCTS_DIR}/librust_lib_qwen_asr.a -framework Accelerate',
  }
  s.swift_version = '5.0'
end
