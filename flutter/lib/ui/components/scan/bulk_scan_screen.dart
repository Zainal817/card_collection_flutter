import 'dart:io';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import '../services/api_service.dart';
import '../widgets/scan_frame.dart';
import '../collection/collection.dart';

class BulkScanScreen extends StatefulWidget {
  final ApiService api;
  final List<CameraDescription> cameras;

  const BulkScanScreen({super.key, required this.api, required this.cameras});

  @override
  State<BulkScanScreen> createState() => _BulkScanScreenState();
}

class _BulkScanScreenState extends State<BulkScanScreen> {
  CameraController? _controller;
  bool _isReady = false;
  bool _isCapturing = false;

  final List<Map<String, dynamic>> _results = [];
  final Set<String> _selectedIds = {};

  @override
  void initState() {
    super.initState();
    if (widget.cameras.isNotEmpty) {
      _controller =
          CameraController(widget.cameras.first, ResolutionPreset.medium);
      _controller!.initialize().then((_) {
        if (mounted) setState(() => _isReady = true);
      });
    }
  }

  Future<void> _capture() async {
    try {
      for (var i = 0; i < 6; i++) {
        setState(() {
          _results.add({
            'id': i.toString(),
            'name': 'Radabra_$i',
            'price': '\$' + (100 + 15 * i).toString(),
            'image_url': null,
            'condition': 'NM',
          });
        });
      }
    } catch (e) {
      debugPrint("Bulk scan error: $e");
    } finally {
      setState(() => _isCapturing = false);
    }
  }

  void _toggleSelect(String id) {
    setState(() {
      if (_selectedIds.contains(id)) {
        _selectedIds.remove(id);
      } else {
        _selectedIds.add(id);
      }
    });
  }

  void _submitSelected() {
    final selectedCards =
        _results.where((c) => _selectedIds.contains(c['id'])).toList();

    if (_results.isEmpty) return;

    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (_) => CollectionScreen(
          scannedCards: selectedCards
              .map((e) => e.map((k, v) => MapEntry(k, v.toString())))
              .toList(),
        ),
      ),
    );
  }

  @override
  void dispose() {
    _controller?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF0A0E21), // Dark theme background
      appBar: AppBar(
        title: const Text("Bulk Card Scanning"),
        backgroundColor: const Color(0xFF1B263B),
        elevation: 0,
        actions: [
          if (_selectedIds.isNotEmpty)
            IconButton(
              icon: const Icon(Icons.check, color: Colors.amber),
              onPressed: _submitSelected,
            ),
        ],
      ),
      body: Stack(
        children: [
          // Camera preview with dim overlay
          Positioned.fill(
            child: _isReady && _controller != null
                ? ColorFiltered(
                    colorFilter: ColorFilter.mode(
                      Colors.black.withOpacity(0.5),
                      BlendMode.darken,
                    ),
                    child: CameraPreview(_controller!),
                  )
                : const Center(child: CircularProgressIndicator()),
          ),

          // Overlay scan frame
          Positioned.fill(
            child: IgnorePointer(
              child: CustomPaint(painter: ScanFramePainter()),
            ),
          ),

          // Results Grid with dark cards
          if (_results.isNotEmpty)
            Container(
              padding: const EdgeInsets.all(16),
              child: GridView.builder(
                itemCount: _results.length,
                gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
                  crossAxisCount: 2,
                  crossAxisSpacing: 12,
                  mainAxisSpacing: 12,
                  childAspectRatio: 3 / 4,
                ),
                itemBuilder: (context, index) {
                  final card = _results[index];
                  final id = card['id'];
                  final isSelected = _selectedIds.contains(id);

                  return GestureDetector(
                    onTap: () => _toggleSelect(id),
                    child: Container(
                      decoration: BoxDecoration(
                        borderRadius: BorderRadius.circular(12),
                        color: const Color(0xFF1B263B),
                        border: Border.all(
                          color:
                              isSelected ? Colors.amber : Colors.transparent,
                          width: 2,
                        ),
                      ),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.center,
                        children: [
                          Expanded(
                            child: ClipRRect(
                              borderRadius:
                                  const BorderRadius.vertical(top: Radius.circular(12)),
                              child: card['image_url'] != null
                                  ? Image.file(
                                      File(card['image_url']),
                                      fit: BoxFit.cover,
                                      width: double.infinity,
                                    )
                                  : const Icon(Icons.style,
                                      size: 80, color: Colors.white54),
                            ),
                          ),
                          const SizedBox(height: 6),
                          Text(
                            card['name'] ?? "Unknown",
                            style: const TextStyle( 
                              fontWeight: FontWeight.bold,
                              color: Colors.white,
                              fontSize: 14,
                            ),
                            overflow: TextOverflow.ellipsis,
                          ),
                          const SizedBox(height: 4),
                          Text(
                            "${card['price'] ?? '0.00'}",
                            style: const TextStyle(
                              color: Colors.amber,
                              fontWeight: FontWeight.w600,
                            ),
                          ),
                          const SizedBox(height: 8),
                        ],
                      ),
                    ),
                  );
                },
              ),
            ),

          // Floating Circular Gradient Button
          Align(
            alignment: Alignment.bottomCenter,
            child: Padding(
              padding: const EdgeInsets.only(bottom: 30),
              child: GestureDetector(
                onTap: _results.isEmpty ? _capture : _submitSelected,
                child: Container(
                  width: 70,
                  height: 70,
                  decoration: BoxDecoration(
                    shape: BoxShape.circle,
                    gradient: const LinearGradient(
                      colors: [Color(0xFF8A2BE2), Color(0xFF1565C0)],
                      begin: Alignment.topLeft,
                      end: Alignment.bottomRight,
                    ),
                  ),
                  child: Icon(
                    _results.isEmpty ? Icons.camera_alt : Icons.check,
                    color: Colors.white,
                    size: 32,
                  ),
                ),
              ),
            ),
          )
        ],
      ),
    );
  }
}
