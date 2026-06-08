package com.example.soundpulse_mobile

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.os.Build

class BootReceiver : BroadcastReceiver() {
    override fun onReceive(context: Context, intent: Intent) {
        if (intent.action == Intent.ACTION_BOOT_COMPLETED || 
            intent.action == "android.intent.action.QUICKBOOT_POWERON") {
            
            val prefs = context.getSharedPreferences("soundpulse", Context.MODE_PRIVATE)
            val active = prefs.getBoolean("monitoring_active", false)
            if (active) {
                val serviceIntent = Intent(context, SoundMonitoringService::class.java).apply {
                    action = "START_MONITORING"
                    putExtra("started_from_boot", true)
                }
                
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                    context.startForegroundService(serviceIntent)
                } else {
                    context.startService(serviceIntent)
                }
            }
        }
    }
}
