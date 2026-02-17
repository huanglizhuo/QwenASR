package com.example.qwen_asr

import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel
import org.mockito.Mockito
import kotlin.test.Test

internal class QwenAsrPluginTest {
    @Test
    fun onMethodCall_getPlatformVersion_returnsExpectedValue() {
        val plugin = QwenAsrPlugin()

        val call = MethodCall("getPlatformVersion", null)
        val mockResult: MethodChannel.Result = Mockito.mock(MethodChannel.Result::class.java)
        plugin.onMethodCall(call, mockResult)

        Mockito.verify(mockResult).success("Android " + android.os.Build.VERSION.RELEASE)
    }
}
