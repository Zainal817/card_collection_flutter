import 'package:flutter/material.dart';
import 'card_detail.dart';

class CollectionScreen extends StatefulWidget {
  final List<Map<String, dynamic>>? scannedCards; // Cards passed from scan
  const CollectionScreen({super.key, this.scannedCards});

  @override
  State<CollectionScreen> createState() => _CollectionScreenState();
}

class _CollectionScreenState extends State<CollectionScreen>
    with SingleTickerProviderStateMixin {
  late TabController _tabController;
  late TextEditingController _searchController;
  String _searchQuery = "";

  late List<Map<String, dynamic>> _cards;

  @override
  void initState() {
    super.initState();
    _tabController = TabController(length: 3, vsync: this);
    _searchController = TextEditingController();

    // Default demo data
    _cards = [
      {
        'id': 'card_1',
        'name': 'Venusaur',
        'condition': 'New Mint',
        'image_url': 'https://via.placeholder.com/80x110.png?text=Venusaur',
        'price': '\$81.23',
      },
      {
        'id': 'card_2',
        'name': 'Mewfup',
        'condition': 'Lightly Played',
        'image_url': 'https://via.placeholder.com/80x110.png?text=Mewfup',
        'price': '\$22.50',
      },
      {
        'id': 'card_3',
        'name': 'Hitmortop',
        'condition': 'New Mint',
        'image_url': 'https://via.placeholder.com/80x110.png?text=Hitmortop',
        'price': '\$11.66',
      },
      {
        'id': 'card_4',
        'name': 'Kleavor',
        'condition': 'Heavily Played',
        'image_url': 'https://via.placeholder.com/80x110.png?text=Kleavor',
        'price': '\$4.06',
      },
    ];

    // Merge scanned cards
    if (widget.scannedCards != null) {
      _cards.addAll(widget.scannedCards!);
    }
  }

  @override
  void dispose() {
    _tabController.dispose();
    _searchController.dispose();
    super.dispose();
  }

  bool _isSearching = false;

  @override
  Widget build(BuildContext context) {
    final filteredCards = _cards
        .where((c) =>
            c['name']!.toLowerCase().contains(_searchQuery.toLowerCase()))
        .toList();

    return Scaffold(
      backgroundColor: const Color(0xFF1B263B),
      appBar: AppBar(
        elevation: 0,
        title: !_isSearching
            ? const Text("My Collection")
            : TextField(
                controller: _searchController,
                autofocus: true,
                style: const TextStyle(color: Colors.white),
                decoration: const InputDecoration(
                  hintText: "Search cards...",
                  border: InputBorder.none,
                  hintStyle: TextStyle(color: Colors.white70),
                ),
                onChanged: (value) {
                  setState(() => _searchQuery = value);
                },
              ),
        centerTitle: true,
        backgroundColor: const Color(0xFF1B263B),
        actions: [
          IconButton(
            icon: Icon(
              _isSearching ? Icons.close : Icons.search,
              color: Colors.amber,
            ),
            onPressed: () {
              setState(() {
                if (_isSearching) {
                  _searchQuery = "";
                  _searchController.clear();
                }
                _isSearching = !_isSearching;
              });
            },
          ),
        ],
        bottom: TabBar(
          controller: _tabController,
          labelColor: Colors.amber,
          unselectedLabelColor: Colors.white70,
          indicatorColor: Colors.amber,
          tabs: const [
            Tab(text: "All"),
            Tab(text: "Favorites"),
            Tab(text: "Unclassified"),
          ],
        ),
      ),
      body: Column(
        children: [
          // Collection header
          Container(
            padding: const EdgeInsets.all(12),
            color: const Color(0xFF1B263B),
            child: Row(
              children: [
                const CircleAvatar(
                  radius: 22,
                  backgroundColor: Colors.amber,
                  child: Text(
                    "G",
                    style: TextStyle(
                      fontWeight: FontWeight.bold,
                      color: Colors.white,
                    ),
                  ),
                ),
                const SizedBox(width: 12),
                Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: const [
                    Text(
                      "Groovy's GMI",
                      style: TextStyle(
                        fontSize: 18,
                        fontWeight: FontWeight.bold,
                        color: Colors.white,
                      ),
                    ),
                    Text(
                      "Personal Collection",
                      style: TextStyle(color: Colors.white70, fontSize: 13),
                    ),
                  ],
                )
              ],
            ),
          ),

          // Tab content
          Expanded(
            child: TabBarView(
              controller: _tabController,
              children: [
                _buildCardList(filteredCards),
                _buildCardList(filteredCards.take(2).toList()),
                _buildCardList([]),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildCardList(List<Map<String, dynamic>> cards) {
    if (cards.isEmpty) {
      return const Center(
        child: Text(
          "No cards found",
          style: TextStyle(color: Colors.white70, fontSize: 16),
        ),
      );
    }

    return ListView.separated(
      physics: const BouncingScrollPhysics(),
      itemCount: cards.length,
      separatorBuilder: (_, __) => const Divider(height: 1, color: Colors.white12),
      itemBuilder: (context, idx) {
        final c = cards[idx];
        return Dismissible(
          key: ValueKey(c['id']),
          direction: DismissDirection.endToStart,
          background: Container(
            color: Colors.redAccent,
            alignment: Alignment.centerRight,
            padding: const EdgeInsets.symmetric(horizontal: 20),
            child: const Icon(Icons.delete, color: Colors.white),
          ),
          onDismissed: (_) {
            setState(() {
              _cards.removeWhere((card) => card['id'] == c['id']);
            });
          },
          child: ListTile(
            tileColor: const Color(0xFF1B263B),
            leading: ClipRRect(
              borderRadius: BorderRadius.circular(6),
              child: Image.network(
                c['image_url']!,
                width: 60,
                height: 80,
                fit: BoxFit.cover,
                errorBuilder: (_, __, ___) => Container(
                  width: 60,
                  height: 80,
                  color: Colors.grey.shade800,
                  alignment: Alignment.center,
                  child: const Icon(Icons.broken_image, color: Colors.white54),
                ),
              ),
            ),
            title: Text(
              c['name']!,
              style: const TextStyle(
                fontWeight: FontWeight.bold,
                fontSize: 16,
                color: Colors.white,
              ),
            ),
            subtitle: Text(
              c['condition']!,
              style: const TextStyle(color: Colors.white70, fontSize: 13),
            ),
            trailing: Text(
              c['price']!,
              style: const TextStyle(
                fontWeight: FontWeight.bold,
                fontSize: 15,
                color: Colors.amber,
              ),
            ),
            onTap: () {
              Navigator.of(context).push(MaterialPageRoute(
                builder: (_) => CardDetailScreen(cardData: c),
              ));
            },
          ),
        );
      },
    );
  }
}
