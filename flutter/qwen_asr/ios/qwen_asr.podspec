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
  s.dependency 'Flutter'
  s.platform = :ios, '13.0'
  s.frameworks = 'Accelerate'

  s.pod_target_xcconfig = { 'DEFINES_MODULE' => 'YES', 'EXCLUDED_ARCHS[sdk=iphonesimulator*]' => 'i386' }
  s.swift_version = '5.0'
end
