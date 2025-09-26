import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import '../services/api_service.dart';
import 'scan_screen.dart';
import 'bulk_scan_screen.dart';

class ScanOptionsPage extends StatelessWidget {
  final ApiService api;
  final List<CameraDescription> cameras;

  const ScanOptionsPage({
    super.key,
    required this.api,
    required this.cameras,
  });

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF0A0E21), // Dark elegant background
      appBar: AppBar(
        title: const Text("Choose Scan Mode"),
        backgroundColor: const Color(0xFF1B263B),
        centerTitle: true,
        elevation: 0,
      ),
      body: Padding(
        padding: const EdgeInsets.all(24),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            // Title
            const Text(
              "Scan Your Cards",
              style: TextStyle(
                fontSize: 26,
                fontWeight: FontWeight.bold,
                color: Colors.white,
              ),
            ),
            const SizedBox(height: 12),
            const Text(
              "Choose a mode to start scanning",
              style: TextStyle(
                fontSize: 16,
                color: Colors.white70,
              ),
            ),
            const SizedBox(height: 50),

            // 🌟 Single Scan Button
            _buildGlowingButton(
              context,
              label: "Single Scan",
              icon: Icons.camera_alt,
              gradient: const LinearGradient(
                colors: [Color(0xFF8A2BE2), Color(0xFF00BFFF)],
                begin: Alignment.topLeft,
                end: Alignment.bottomRight,
              ),
              onTap: () {
                Navigator.push(
                  context,
                  MaterialPageRoute(
                    builder: (_) => ScanScreen(api: api, cameras: cameras),
                  ),
                );
              },
            ),
            const SizedBox(height: 24),

            // 🌟 Bulk Scan Button
            _buildGlowingButton(
              context,
              label: "Bulk Scan",
              icon: Icons.collections,
              gradient: const LinearGradient(
                colors: [Color(0xFFFF6B6B), Color(0xFFFFD93D)],
                begin: Alignment.topLeft,
                end: Alignment.bottomRight,
              ),
              onTap: () {
                Navigator.push(
                  context,
                  MaterialPageRoute(
                    builder: (_) =>
                        BulkScanScreen(api: api, cameras: cameras),
                  ),
                );
              },
            ),
          ],
        ),
      ),
    );
  }

  /// 🔥 Reusable glowing gradient button
  Widget _buildGlowingButton(
    BuildContext context, {
    required String label,
    required IconData icon,
    required Gradient gradient,
    required VoidCallback onTap,
  }) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        width: double.infinity,
        padding: const EdgeInsets.symmetric(vertical: 20, horizontal: 16),
        decoration: BoxDecoration(
          gradient: gradient,
          borderRadius: BorderRadius.circular(18),
        ),
        child: Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(icon, size: 26, color: Colors.white),
            const SizedBox(width: 12),
            Text(
              label.toUpperCase(),
              style: const TextStyle(
                fontSize: 20,
                fontWeight: FontWeight.bold,
                letterSpacing: 1.5,
                color: Colors.white,
              ),
            ),
          ],
        ),
      ),
    );
  }
}
