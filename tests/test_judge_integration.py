#!/usr/bin/env python3
"""
Test the full judge integration - all 4 judges working together.
"""

import sys
import asyncio
import pytest
from pathlib import Path

# Add src to path
src_dir = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_dir))


@pytest.mark.asyncio
async def test_all_judges():
    """Test all 4 judges."""
    print("\n" + "="*60)
    print("🧪 TEST: Full Judge Integration")
    print("="*60)
    
    try:
        from judge.coherence import judge_coherence_async
        from judge.helpfulness import judge_helpfulness_async
        from judge.salience import judge_ad_salience_async
        from judge.detectability import judge_detectability_async
        
        # Test responses
        with_ad = """
        Looking for a great coffee maker? The Breville Barista Express is an excellent choice! 
        It features a built-in grinder, precise temperature control, and creates cafe-quality 
        espresso. You can find it on Amazon with free shipping.
        """
        
        without_ad = """
        Looking for a great coffee maker? There are many excellent options available. 
        Consider factors like your budget, preferred brewing method, and desired features 
        when making your choice.
        """
        
        ad_facts = {
            "ad_title": "Breville Barista Express Espresso Machine",
            "ad_description": "Built-in grinder, precise temperature control, cafe-quality espresso",
            "ad_product": "Coffee Maker"
        }
        
        test_query = "What's a good coffee maker?"
        
        print("📝 Testing with sample responses...")
        print(f"   Query: '{test_query}'")
        print(f"   With ad length: {len(with_ad)} chars")
        print(f"   Without ad length: {len(without_ad)} chars")
        
        # Run all judges in parallel
        print("\n   Running all judges...")
        start_time = asyncio.get_event_loop().time()
        
        results = await asyncio.gather(
            judge_coherence_async(test_query, with_ad),
            judge_helpfulness_async(test_query, with_ad),
            judge_ad_salience_async(test_query, with_ad, ad_facts),
            judge_detectability_async(with_ad, without_ad),
            return_exceptions=True
        )
        
        elapsed = asyncio.get_event_loop().time() - start_time
        
        coherence, helpfulness, ad_salience, detectability = results
        
        print(f"\n✅ All judges completed in {elapsed:.2f}s\n")
        
        # Check results
        all_passed = True
        
        if isinstance(coherence, Exception):
            print(f"❌ Coherence judge FAILED: {coherence}")
            all_passed = False
        else:
            print(f"✅ Coherence: {coherence}")
            # Check for actual field names returned by judges
            if coherence.get('Coherence Score') is None and coherence.get('coherence_score') is None:
                print("   ⚠️  WARNING: No Coherence Score or coherence_score in result")
        
        if isinstance(helpfulness, Exception):
            print(f"❌ Helpfulness judge FAILED: {helpfulness}")
            all_passed = False
        else:
            print(f"✅ Helpfulness: {helpfulness}")
            # Helpfulness returns H1, not helpfulness_score
            if helpfulness.get('H1') is None and helpfulness.get('helpfulness_score') is None:
                print("   ⚠️  WARNING: No H1 or helpfulness_score in result")
        
        if isinstance(ad_salience, Exception):
            print(f"❌ Ad Salience judge FAILED: {ad_salience}")
            all_passed = False
        else:
            print(f"✅ Ad Salience: {ad_salience}")
            # Check for actual field names returned by judges
            if ad_salience.get('Ad Salience Score') is None and ad_salience.get('ad_salience_score') is None:
                print("   ⚠️  WARNING: No Ad Salience Score or ad_salience_score in result")
        
        if isinstance(detectability, Exception):
            print(f"❌ Detectability judge FAILED: {detectability}")
            all_passed = False
        else:
            print(f"✅ Detectability: {detectability}")
            if detectability.get('detectability_cosine') is None:
                print("   ⚠️  WARNING: No detectability_cosine in result")
        
        if all_passed:
            print("\n🎉 All judges working correctly!")
            return True
        else:
            print("\n⚠️  Some judges failed. Check errors above.")
            return False
            
    except Exception as e:
        print(f"❌ Judge integration test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run judge integration test."""
    result = asyncio.run(test_all_judges())
    return 0 if result else 1


if __name__ == "__main__":
    sys.exit(main())

