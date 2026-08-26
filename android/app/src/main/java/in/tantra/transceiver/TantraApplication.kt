package in.tantra.transceiver

import android.app.Application
import in.tantra.transceiver.runtime.TantraRuntime

class TantraApplication : Application() {
    val runtime: TantraRuntime by lazy { TantraRuntime(this) }
    override fun onTerminate() { runtime.close(); super.onTerminate() }
}
