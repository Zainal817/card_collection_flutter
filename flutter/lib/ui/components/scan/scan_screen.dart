import 'dart:io';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import '../services/api_service.dart';
import '../widgets/scan_frame.dart';
import '../collection/card_detail.dart';

class ScanScreen extends StatefulWidget {
  final ApiService api;
  final List<CameraDescription> cameras;
  const ScanScreen({super.key, required this.api, required this.cameras});

  @override
  State<ScanScreen> createState() => _ScanScreenState();
}

class _ScanScreenState extends State<ScanScreen>
    with SingleTickerProviderStateMixin {
  CameraController? _controller;
  bool _isReady = false;
  Map<String, dynamic>? _result = {
    'id': 'charizard_1',
    'name': 'Charizard',
    'image_url': 'https://via.placeholder.com/400x600.png?text=Charizard',
    'prices': [3.0, 31.77, 107.23, 92.0, 110.0, 95.0],
    'status': 'Available',
  };

  late final AnimationController _glowController;
  late final Animation<double> _glowAnimation;

  @override
  void initState() {
    super.initState();
    _initCamera();

    // glowing button animation
    _glowController =
        AnimationController(vsync: this, duration: const Duration(seconds: 2))
          ..repeat(reverse: true);

    _glowAnimation =
        Tween<double>(begin: 0.95, end: 1.05).animate(CurvedAnimation(
      parent: _glowController,
      curve: Curves.easeInOut,
    ));
  }

  Future<void> _initCamera() async {
    if (widget.cameras.isEmpty) return;

    _controller = CameraController(
      widget.cameras.first,
      ResolutionPreset.medium,
    );

    await _controller!.initialize();
    if (!mounted) return;
    setState(() => _isReady = true);
  }

  Future<void> _capture() async {

    print(await widget.api.identifyCard(File(picture.path)));
    if (_controller == null || !_controller!.value.isInitialized) return;

    final picture = await _controller!.takePicture();

    // 🔹 Call API (replace with your service)
    final resp = await widget.api.identifyCard(File(picture.path));

    setState(() => _result = resp);
  }

  @override
  void dispose() {
    _controller?.dispose();
    _glowController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF0A0E21),
      appBar: AppBar(
        title: const Text("Scan Card"),
        backgroundColor: const Color(0xFF1B263B),
      ),
      body: Stack(
        children: [
          // camera preview
          Positioned.fill(
            child: _isReady && _controller != null
                ? CameraPreview(_controller!)
                : const Center(child: CircularProgressIndicator()),
          ),

          // scan frame overlay
          Positioned.fill(
            child: IgnorePointer(child: CustomPaint(painter: ScanFramePainter())),
          ),


          // result overlay
          if (_result != null)
            Positioned(
              bottom: 120,
              left: 20,
              right: 20,
              child: _ResultOverlay(
                result: _result!,
                onOpenDetail: () {
                  Navigator.push(
                    context,
                    MaterialPageRoute(
                      builder: (_) => CardDetailScreen(cardData: _result!),
                    ),
                  );
                },
              ),
            ),

          // glowing SCAN button
          Align(
            alignment: Alignment.bottomCenter,
            child: Padding(
              padding: const EdgeInsets.only(bottom: 36),
              child: ScaleTransition(
                scale: _glowAnimation,
                child: GestureDetector(
                  onTap: _capture,
                  child: Container(
                    padding:
                        const EdgeInsets.symmetric(horizontal: 40, vertical: 16),
                    decoration: BoxDecoration(
                      gradient: const LinearGradient(
                        colors: [Color(0xFF7B1FA2), Color(0xFF1565C0)],
                        begin: Alignment.topLeft,
                        end: Alignment.bottomRight,
                      ),
                      borderRadius: BorderRadius.circular(32),
                    ),
                    child: const Text(
                      "SCAN",
                      style: TextStyle(
                        fontSize: 20,
                        fontWeight: FontWeight.bold,
                        color: Colors.white,
                        letterSpacing: 2,
                      ),
                    ),
                  ),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

/// ---------- Result overlay ----------
class _ResultOverlay extends StatelessWidget {
  final Map<String, dynamic> result;
  final VoidCallback onOpenDetail;
  const _ResultOverlay({required this.result, required this.onOpenDetail});

  @override
  Widget build(BuildContext context) {
    final isLocalFile =
        result['image_url'] != null && File(result['image_url']).existsSync();

    return Container(
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(16),
        boxShadow: [
          BoxShadow(
            color: Colors.black12,
            blurRadius: 10,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      padding: const EdgeInsets.all(12),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(children: [
            Material(
              elevation: 5,
              borderRadius: BorderRadius.circular(6),
              child: isLocalFile
                  ? Image.file(File(result['image_url']),
                      width: 60, height: 80, fit: BoxFit.cover)
                  : Image.network(result['image_url'] ?? "",
                      width: 60, height: 80, fit: BoxFit.cover,
                      errorBuilder: (_, __, ___) => Container(
                            width: 60,
                            height: 80,
                            color: Colors.grey.shade300,
                            child: const Icon(Icons.broken_image),
                          )),
            ),
            const SizedBox(width: 12),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(result['name'] ?? "Unknown",
                      style: const TextStyle(
                        fontSize: 18,
                        fontWeight: FontWeight.bold,
                        color: Colors.black87,
                      )),
                  const SizedBox(height: 6),
                  Container(
                    padding: const EdgeInsets.symmetric(
                        horizontal: 10, vertical: 6),
                    decoration: BoxDecoration(
                      color: Colors.blueAccent,
                      borderRadius: BorderRadius.circular(12),
                    ),
                    child: Text(result['status'] ?? "N/A",
                        style: const TextStyle(
                            color: Colors.white,
                            fontWeight: FontWeight.bold)),
                  ),
                ],
              ),
            ),
            IconButton(
              onPressed: onOpenDetail,
              icon: const Icon(Icons.chevron_right, color: Colors.black87),
            ),
          ]),
          const SizedBox(height: 10),
          if (result['prices'] != null)
            SizedBox(
              height: 60,
              child: _PriceSparkline(
                  prices: List<double>.from(result['prices'])),
            ),
          const SizedBox(height: 8),
          SizedBox(
            width: double.infinity,
            child: ElevatedButton(
              onPressed: onOpenDetail,
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.blueAccent,
              ),
              child: const Text('View details',
              style: TextStyle(
                fontSize: 16,
                fontWeight: FontWeight.bold,
                color: Colors.white,
              )),
            ),
          ),
        ],
      ),
    );
  }
}

/// ---------- Price sparkline ----------
class _PriceSparkline extends StatelessWidget {
  final List<double> prices;
  const _PriceSparkline({required this.prices});

  @override
  Widget build(BuildContext context) {
    if (prices.isEmpty) return const SizedBox.shrink();
    final maxPrice = prices.reduce((a, b) => a > b ? a : b);
    return Row(
      crossAxisAlignment: CrossAxisAlignment.end,
      children: prices.map((p) {
        final heightFactor = (p / maxPrice).clamp(0.05, 1.0);
        return Expanded(
          child: Container(
            margin: const EdgeInsets.symmetric(horizontal: 2),
            height: 60 * heightFactor,
            decoration: BoxDecoration(
              color: Colors.blueAccent,
              borderRadius: BorderRadius.circular(4),
            ),
          ),
        );
      }).toList(),
    );
  }
}
