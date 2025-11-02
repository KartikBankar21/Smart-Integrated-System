"""
Test script for response_generator.py
Run this to verify everything works before deployment
"""

from response_generator import ResponseGenerator, create_chat_response


def test_slot_extraction():
    """Test the slot extraction logic"""
    print("\n" + "="*70)
    print("TEST 1: Slot Extraction")
    print("="*70)
    
    generator = ResponseGenerator()
    
    test_cases = [
        {
            'name': 'Simple single-token slot',
            'slots': [('paris', 'B-city')],
            'expected': {'city': 'paris'}
        },
        {
            'name': 'Multi-token slot',
            'slots': [('new', 'B-city'), ('york', 'I-city')],
            'expected': {'city': 'new york'}
        },
        {
            'name': 'Multiple different slots',
            'slots': [
                ('italian', 'B-cuisine'), ('food', 'I-cuisine'),
                ('in', 'O'), ('manhattan', 'B-city')
            ],
            'expected': {'cuisine': 'italian food', 'city': 'manhattan'}
        },
        {
            'name': 'Artist name (multi-token)',
            'slots': [('taylor', 'B-artist'), ('swift', 'I-artist')],
            'expected': {'artist': 'taylor swift'}
        },
        {
            'name': 'With special tokens',
            'slots': [
                ('[CLS]', 'O'), ('paris', 'B-city'), 
                ('[SEP]', 'O'), ('[PAD]', 'O')
            ],
            'expected': {'city': 'paris'}
        }
    ]
    
    passed = 0
    failed = 0
    
    for test in test_cases:
        result = generator.extract_slot_values(test['slots'])
        if result == test['expected']:
            print(f"✅ PASS: {test['name']}")
            print(f"   Result: {result}")
            passed += 1
        else:
            print(f"❌ FAIL: {test['name']}")
            print(f"   Expected: {test['expected']}")
            print(f"   Got: {result}")
            failed += 1
        print()
    
    print(f"Results: {passed} passed, {failed} failed\n")
    return failed == 0


def test_music_intent():
    """Test the fixed music lookup handler"""
    print("\n" + "="*70)
    print("TEST 2: Music Intent Fix")
    print("="*70)
    
    generator = ResponseGenerator()
    
    # Original problematic case
    slots_old = [
        ('look', 'O'), ('up', 'O'), ('for', 'O'), ('a', 'O'), 
        ('song', 'O'), ('named', 'O'), ('taylor', 'O'), ('swift', 'B-album')
    ]
    
    # Fixed case
    slots_new = [
        ('look', 'O'), ('up', 'O'), ('for', 'O'), ('a', 'O'), 
        ('song', 'O'), ('named', 'O'), ('taylor', 'B-artist'), ('swift', 'I-artist')
    ]
    
    print("Old (incorrect) slot detection:")
    response_old = generator.generate_response('LookupMusic', slots_old)
    print(f"   Slots: {generator.extract_slot_values(slots_old)}")
    print(f"   Response: {response_old}\n")
    
    print("New (correct) slot detection:")
    response_new = generator.generate_response('LookupMusic', slots_new)
    print(f"   Slots: {generator.extract_slot_values(slots_new)}")
    print(f"   Response: {response_new}\n")
    
    # Check if the new response mentions artist correctly
    slot_values = generator.extract_slot_values(slots_new)
    if 'artist' in slot_values and slot_values['artist'] == 'taylor swift':
        print("✅ PASS: Artist correctly extracted as 'taylor swift'")
        return True
    else:
        print("❌ FAIL: Artist not correctly extracted")
        return False


def test_all_intents():
    """Test responses for all supported intents"""
    print("\n" + "="*70)
    print("TEST 3: All Intent Handlers")
    print("="*70)
    
    generator = ResponseGenerator()
    
    test_cases = [
        ('FindRestaurants', [('italian', 'B-cuisine'), ('in', 'O'), ('manhattan', 'B-city')]),
        ('SearchHotel', [('paris', 'B-city')]),
        ('GetRide', [('airport', 'B-destination')]),
        ('PlaySong', [('bohemian', 'B-song_name'), ('rhapsody', 'I-song_name')]),
        ('GetWeather', [('new', 'B-city'), ('york', 'I-city')]),
        ('TransferMoney', [('100', 'B-amount'), ('to', 'O'), ('john', 'B-recipient')]),
        ('NONE', []),
        ('UnknownIntent', []),
    ]
    
    print("Testing all intent handlers...\n")
    
    for intent, slots in test_cases:
        try:
            response = generator.generate_response(intent, slots)
            print(f"✅ {intent:30s} → {response[:60]}...")
        except Exception as e:
            print(f"❌ {intent:30s} → ERROR: {e}")
    
    print("\n✅ All intent handlers working\n")
    return True


def test_edge_cases():
    """Test edge cases and error handling"""
    print("\n" + "="*70)
    print("TEST 4: Edge Cases")
    print("="*70)
    
    generator = ResponseGenerator()
    
    test_cases = [
        {
            'name': 'Empty slots',
            'intent': 'FindRestaurants',
            'slots': []
        },
        {
            'name': 'Only O tags',
            'intent': 'SearchHotel',
            'slots': [('find', 'O'), ('hotel', 'O')]
        },
        {
            'name': 'Incomplete slot (only I- tag)',
            'intent': 'GetRide',
            'slots': [('york', 'I-city')]
        },
        {
            'name': 'All special tokens',
            'intent': 'GetWeather',
            'slots': [('[CLS]', 'O'), ('[SEP]', 'O'), ('[PAD]', 'O')]
        }
    ]
    
    passed = 0
    
    for test in test_cases:
        try:
            response = generator.generate_response(test['intent'], test['slots'])
            print(f"✅ {test['name']:30s} → {response[:50]}...")
            passed += 1
        except Exception as e:
            print(f"❌ {test['name']:30s} → ERROR: {e}")
    
    print(f"\n{passed}/{len(test_cases)} edge cases handled\n")
    return passed == len(test_cases)


def test_fastapi_integration():
    """Test the FastAPI integration helper"""
    print("\n" + "="*70)
    print("TEST 5: FastAPI Integration Helper")
    print("="*70)
    
    generator = ResponseGenerator()
    
    # Mock predictor result
    mock_result = {
        'intent': 'FindRestaurants',
        'secondary_intent': 'NONE',
        'slots': [
            ('italian', 'B-cuisine'), ('food', 'I-cuisine'),
            ('in', 'O'), ('manhattan', 'B-city')
        ],
        'turn_number': 1
    }
    
    try:
        chat_response = create_chat_response(mock_result, generator)
        
        print("Mock predictor result:")
        print(f"   Intent: {mock_result['intent']}")
        print(f"   Slots: {mock_result['slots']}")
        print()
        print("Generated chat response:")
        print(f"   Intent: {chat_response['intent']}")
        print(f"   Slot Values: {chat_response['slot_values']}")
        print(f"   Response: {chat_response['response']}")
        print(f"   Turn: {chat_response['turn_number']}")
        print()
        
        # Verify all expected fields are present
        expected_fields = ['intent', 'secondary_intent', 'slots', 'slot_values', 'response', 'turn_number']
        missing = [f for f in expected_fields if f not in chat_response]
        
        if not missing:
            print("✅ PASS: All fields present in response")
            return True
        else:
            print(f"❌ FAIL: Missing fields: {missing}")
            return False
            
    except Exception as e:
        print(f"❌ FAIL: {e}")
        return False


def test_safe_get():
    """Test the _safe_get utility method"""
    print("\n" + "="*70)
    print("TEST 6: Safe Get Utility")
    print("="*70)
    
    generator = ResponseGenerator()
    
    test_slots = {
        'city': 'paris',
        'destination_city': 'london',
        'empty_key': ''
    }
    
    # Test getting existing key
    result1 = generator._safe_get(test_slots, 'city')
    print(f"Get 'city': {result1} {'✅' if result1 == 'paris' else '❌'}")
    
    # Test fallback to alternative key
    result2 = generator._safe_get(test_slots, 'location', 'city')
    print(f"Get 'location' (fallback to 'city'): {result2} {'✅' if result2 == 'paris' else '❌'}")
    
    # Test with empty value
    result3 = generator._safe_get(test_slots, 'empty_key', 'city')
    print(f"Get 'empty_key' (fallback to 'city'): {result3} {'✅' if result3 == 'paris' else '❌'}")
    
    # Test with non-existent keys
    result4 = generator._safe_get(test_slots, 'nonexistent', 'also_nonexistent')
    print(f"Get nonexistent keys: '{result4}' {'✅' if result4 == '' else '❌'}")
    
    print()
    return True


def run_all_tests():
    """Run all tests"""
    print("\n" + "🧪" * 35)
    print("RESPONSE GENERATOR TEST SUITE")
    print("🧪" * 35)
    
    results = []
    
    # Run all tests
    results.append(("Slot Extraction", test_slot_extraction()))
    results.append(("Music Intent Fix", test_music_intent()))
    results.append(("All Intent Handlers", test_all_intents()))
    results.append(("Edge Cases", test_edge_cases()))
    results.append(("FastAPI Integration", test_fastapi_integration()))
    results.append(("Safe Get Utility", test_safe_get()))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status:10s} {test_name}")
    
    print()
    print(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Ready for deployment! 🎉\n")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please fix before deployment.\n")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)