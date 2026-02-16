Pod::Spec.new do |s|
  s.name             = 'qasr'
  s.version          = '0.0.1'
  s.summary          = 'CPU-only Qwen3-ASR inference for Flutter.'
  s.description      = 'Flutter plugin wrapping q-asr, a pure-Rust CPU-only Qwen3-ASR engine.'
  s.homepage         = 'https://github.com/user/q-asr'
  s.license          = { :file => '../LICENSE' }
  s.author           = { 'q-asr' => 'noreply@example.com' }
  s.source           = { :path => '.' }
  s.source_files = 'Classes/**/*'
  s.dependency 'FlutterMacOS'
  s.platform = :osx, '10.15'
  s.frameworks = 'Accelerate'

  s.pod_target_xcconfig = { 'DEFINES_MODULE' => 'YES' }
  s.swift_version = '5.0'
end
