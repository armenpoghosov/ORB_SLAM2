// g2o - General Graph Optimization
// Copyright (C) 2011 G. Grisetti, R. Kuemmerle, W. Burgard
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright
//   notice, this list of conditions and the following disclaimer in the
//   documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
// IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
// TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
// PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
// TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "cache.h"
#include "optimizable_graph.h"
#include "factory.h"

#include <iostream>

namespace g2o
{

OptimizableGraph::Vertex* Cache::vertex()
{ 
    CacheContainer* container_ = container();
    return container_ != nullptr ? container_->vertex() : nullptr; 
}

OptimizableGraph* Cache::graph()
{
    CacheContainer* container_ = container();
    return container_ != nullptr ? container_->graph() : nullptr;
}

Cache::CacheKey Cache::key() const
{
    Factory* factory = Factory::instance();
    return CacheKey(factory->tag(this), _parameters);
}
  
void Cache::update()
{
    if (!_updateNeeded)
        return;

    for (Cache* c : _parentCaches)
        c->update();

    updateImpl();
    _updateNeeded = false;
}

Cache* Cache::installDependency(std::string const& type_, std::vector<int> const& parameterIndices)
{
    ParameterVector pv(parameterIndices.size());

    for (size_t i = 0; i < parameterIndices.size(); ++i)
    {
        if (parameterIndices[i] < 0 || parameterIndices[i] >= (int)_parameters.size())
            return 0;

        pv[i] = _parameters[parameterIndices[i]];
    }

    CacheContainer* container_ = container();
    if (container_ == nullptr)
        return 0;

    CacheKey k(type_, pv);
    Cache* c = container_->findCache(k);
    if (c == nullptr)
        c = container_->createCache(k);

    if (c != nullptr)
        _parentCaches.push_back(c);

    return c;
}

Cache* CacheContainer::createCache(Cache::CacheKey const& key)
{
    Factory* f = Factory::instance();

    HyperGraph::HyperGraphElement* e = f->construct(key.type());
    if (!e)
    {
        std::cerr << __PRETTY_FUNCTION__ << std::endl;
        std::cerr << "fatal error in creating cache of type " << key.type() << std::endl;
        return nullptr;
    }

    Cache* c = dynamic_cast<Cache*>(e);
    if (!c)
    {
        std::cerr << __PRETTY_FUNCTION__ << std::endl;
        std::cerr << "fatal error in creating cache of type " << key.type() << std::endl;
        return nullptr;
    }

    c->_container = this;
    c->_parameters = key._parameters;

    if (c->resolveDependancies())
    {
        emplace(key, c);
        c->update();
        return c;
    } 

    return nullptr;
}
  
void CacheContainer::update()
{
    for (auto const& pair : *this)
        pair.second->update();
}

void CacheContainer::setUpdateNeeded(bool needUpdate)
{
    for (auto const& pair : *this)
        pair.second->_updateNeeded = needUpdate;
}

CacheContainer::~CacheContainer()
{
    for (auto const& pair : *this)
        delete pair.second;
}

} // end namespace
